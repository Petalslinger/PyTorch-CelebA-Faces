"""把 Kaggle 的 CelebA 压缩包（jpg）预处理成训练用的 .npy 数组。

为什么先转 npy：
  * CelebA 有 20 万张 jpg，每次训练都解码 jpg 会浪费大量 CPU；
  * 统一尺寸（默认 64x64）后一个 split 只有 0.5~2 GB，可以 mmap 读取；
  * 训练脚本就只需要 numpy + torch，不依赖 torchvision 的 ImageFolder。

用法
----
    # 1) 从 zip 直接转换（自动识别 img_align_celeba/img_align_celeba/*.jpg）
    python prepare_data.py --zip archive.zip --out data/celeba64 --size 64

    # 2) 已经是解压好的文件夹
    python prepare_data.py --src ./img_align_celeba --out data/celeba64 --size 64

    # 3) 先小规模试跑（只取 5000 张训练图）
    python prepare_data.py --zip archive.zip --out data/celeba64_5k --size 64 --max-images 5000

输出目录结构::

    data/celeba64/
        train.npy     # [N,3,S,S] uint8
        val.npy
        test.npy
        meta.json     # 尺寸、数量、划分来源、预处理参数
        preview.png   # 64 张训练样本预览，检查预处理是否正常
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


# --------------------------------------------------------------------------- #
# 单张图片的预处理（子进程里执行，必须是模块级函数）
# --------------------------------------------------------------------------- #


_ZIP_CACHE: Dict[str, zipfile.ZipFile] = {}


def _get_zip(path: str) -> zipfile.ZipFile:
    """每个子进程只打开一次 zip，否则 20 万次重新解析中央目录会非常慢。"""
    z = _ZIP_CACHE.get(path)
    if z is None:
        z = zipfile.ZipFile(path)
        _ZIP_CACHE[path] = z
    return z


def _read_image(src: str, name: str, is_zip: bool) -> Image.Image:
    if is_zip:
        with _get_zip(src).open(name) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    return Image.open(os.path.join(src, name)).convert("RGB")


def _center_square_resize(im: Image.Image, size: int) -> np.ndarray:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    # 放大用 bicubic，缩小用 box/antialias，视觉上更干净
    resample = Image.BICUBIC if size > side else Image.LANCZOS
    im = im.resize((size, size), resample)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)  # HWC -> CHW


def _worker(task: Tuple[str, str, int, bool]) -> Optional[np.ndarray]:
    src, name, size, is_zip = task
    try:
        im = _read_image(src, name, is_zip)
        return _center_square_resize(im, size)
    except Exception as e:  # 单张损坏不应中断整体转换
        print(f"[warn] 跳过 {name}: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# 收集图片列表 & 官方划分
# --------------------------------------------------------------------------- #


def _list_from_zip(zip_path: str) -> List[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXTS]
    if not names:
        raise RuntimeError(f"{zip_path} 里没有找到图片")
    # Kaggle 版本是 img_align_celeba/img_align_celeba/000001.jpg
    if len({Path(n).parent for n in names}) > 1:
        print(f"[info] zip 内有多个图片目录，将全部使用（共 {len(names)} 张）")
    return sorted(names)


def _list_from_dir(src: str) -> List[str]:
    files = []
    for p in Path(src).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            files.append(str(p.relative_to(src)).replace("\\", "/"))
    if not files:
        raise RuntimeError(f"{src} 下没有找到图片")
    return sorted(files)


def _load_partition(zip_path: Optional[str], src: Optional[str]) -> Dict[str, int]:
    """读取 list_eval_partition.csv: image_id, partition (0=train,1=val,2=test)。"""
    candidates: List[str] = []
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            candidates = [n for n in z.namelist() if n.lower().endswith("list_eval_partition.csv")]
        if candidates:
            with zipfile.ZipFile(zip_path) as z:
                text = z.read(candidates[0]).decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            return {row["image_id"]: int(row["partition"]) for row in reader}
    if src:
        for cand in Path(src).rglob("list_eval_partition.csv"):
            with open(cand, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return {row["image_id"]: int(row["partition"]) for row in reader}
    return {}


def _stable_hash(s: str) -> int:
    """跨进程/跨机器稳定的哈希（Python 内置 hash 对字符串是随机加盐的）。"""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _split_of(name: str, partition: Dict[str, int]) -> str:
    """优先用官方划分；缺失时按 8:1:1 的稳定哈希兜底。"""
    stem = Path(name).stem
    if stem in partition:
        return SPLIT_NAMES.get(partition[stem], "train")
    bucket = int(stem) % 10 if stem.isdigit() else _stable_hash(stem) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _run_tasks(tasks, workers: int) -> List[Optional[np.ndarray]]:
    """按顺序返回每张图的结果；多进程不可用时自动退回单进程。"""
    if workers <= 1:
        return [_worker(t) for t in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # map 保持输入顺序，可以直接按长度切片回填
            return list(ex.map(_worker, tasks, chunksize=64))
    except (OSError, PermissionError, NotImplementedError) as e:
        print(f"[warn] 多进程不可用（{e}），退回单进程，速度会慢一些", file=sys.stderr)
        return [_worker(t) for t in tasks]


def main() -> None:
    ap = argparse.ArgumentParser(description="CelebA -> npy 预处理")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", type=str, help="Kaggle 下载的 archive.zip")
    g.add_argument("--src", type=str, help="已解压的图片目录")
    ap.add_argument("--out", type=str, required=True, help="输出目录")
    ap.add_argument("--size", type=int, default=64, help="输出分辨率，论文里 64 最常用")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--max-images", type=int, default=0, help=">0 时只取前 N 张（快速试跑）")
    ap.add_argument("--limit-per-split", type=int, default=0, help=">0 时每个 split 只保留这么多")
    ap.add_argument("--preview", type=int, default=64, help="生成预览图的数量，0 表示不生成")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_zip = bool(args.zip)
    src = str(args.zip) if is_zip else str(args.src)

    print(f"[1/4] 扫描图片列表 ...")
    names = _list_from_zip(src) if is_zip else _list_from_dir(src)
    print(f"      共 {len(names)} 张图片")

    partition = _load_partition(args.zip, args.src)
    print(f"[2/4] 官方划分表: {'已找到' if partition else '未找到（将按 8:1:1 兜底划分）'}"
          f"（{len(partition)} 条）")

    # 按 split 分组，并保证顺序稳定
    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for n in names:
        buckets[_split_of(n, partition)].append(n)

    if args.max_images > 0:
        keep = min(args.max_images, len(names))
        # 保持各 split 比例，先把总量压到 keep
        ratio = keep / len(names)
        for k in buckets:
            buckets[k] = buckets[k][: max(1, int(len(buckets[k]) * ratio))]
        print(f"      限定总样本数 {keep}，实际保留 "
              f"{ {k: len(v) for k, v in buckets.items()} }")
    if args.limit_per_split > 0:
        for k in buckets:
            buckets[k] = buckets[k][: args.limit_per_split]

    tasks = []
    for split in ("train", "val", "test"):
        tasks += [(src, n, args.size, is_zip) for n in buckets[split]]

    print(f"[3/4] 解码 + 居中裁剪 + 缩放到 {args.size}x{args.size}（{args.workers} 进程）...")
    results = _run_tasks(tasks, args.workers)
    arrays: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    failed = 0
    pos = 0
    for split in ("train", "val", "test"):
        n = len(buckets[split])
        chunk = results[pos:pos + n]
        pos += n
        for arr in chunk:
            if arr is None:
                failed += 1
            else:
                arrays[split].append(arr)

    for split in ("train", "val", "test"):
        if not arrays[split]:
            print(f"[warn] {split} 为空，跳过")
            continue
        arr = np.stack(arrays[split], axis=0)
        path = out_dir / f"{split}.npy"
        np.save(path, arr)
        print(f"      {split}.npy  {tuple(arr.shape)}  {path.stat().st_size / 1024 ** 3:.2f} GB")

    meta = {
        "source": src,
        "source_type": "zip" if is_zip else "dir",
        "image_size": args.size,
        "layout": "NCHW uint8, 0-255",
        "preprocess": "center-square-crop -> resize -> RGB",
        "split_source": "list_eval_partition.csv (official)" if partition else "fallback 8:1:1",
        "counts": {k: len(v) for k, v in arrays.items()},
        "failed": failed,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      meta.json 已写入: {json.dumps(meta['counts'])}")

    if args.preview > 0 and arrays["train"]:
        print("[4/4] 生成预览图 ...")
        _save_preview(arrays["train"], out_dir / "preview.png", args.preview)
        print(f"      预览图: {out_dir / 'preview.png'}")

    print("完成。训练命令示例:")
    print(f"  python train_gan.py --data {out_dir} --out runs/gan_{args.size}")
    print(f"  python train_vae.py --data {out_dir} --out runs/vae_{args.size}")


def _save_preview(arrays: List[np.ndarray], path: Path, count: int) -> None:
    from PIL import Image as I

    random.seed(0)
    pick = random.sample(range(len(arrays)), min(count, len(arrays)))
    nrow = 8
    s = arrays[pick[0]].shape[-1]
    ncol = (len(pick) + nrow - 1) // nrow
    canvas = np.zeros((ncol * s, nrow * s, 3), dtype=np.uint8)
    for i, idx in enumerate(pick):
        r, c = divmod(i, nrow)
        canvas[r * s:(r + 1) * s, c * s:(c + 1) * s] = arrays[idx].transpose(1, 2, 0)
    I.fromarray(canvas).save(path)


if __name__ == "__main__":
    main()
