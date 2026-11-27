"""CelebA 数据集读取。

prepare_data.py 会把 jpg 解码、居中裁剪、缩放后存成 uint8 的 .npy
（shape = [N, 3, S, S]，取值 0~255）。这里用 np.load(mmap_mode='r')
按需读取，不必把几个 G 的图片一次性读进内存。

返回给模型的数据统一归一化到 [-1, 1]，GAN 与 VAE 共用同一套预处理。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CelebASplit(Dataset):
    """单个 split（train/val/test）的 CelebA 数组数据集。"""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
        flip_prob: float = 0.5,
        cache_ram: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.flip_prob = flip_prob

        self.npy_path = self.root / f"{split}.npy"
        if not self.npy_path.exists():
            raise FileNotFoundError(
                f"找不到 {self.npy_path}，请先运行: python prepare_data.py --zip archive.zip --out {self.root}"
            )

        # mmap_mode='c' 是 copy-on-write：不占内存、又能得到可写数组，
        # 避免 torch.from_numpy 对只读数组报 UserWarning。
        self.data = np.load(self.npy_path, mmap_mode=None if cache_ram else "c")
        if self.data.ndim != 4 or self.data.shape[1] != 3:
            raise ValueError(f"{self.npy_path} 形状异常: {self.data.shape}，期望 [N, 3, S, S]")
        self.size = int(self.data.shape[-1])

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        """统一返回 (image[3,S,S] in [-1,1], index)，方便反查真实图。"""
        img = self.data[index]  # uint8 [3, S, S]

        if self.augment and np.random.rand() < self.flip_prob:
            img = img[:, :, ::-1]  # 水平翻转（人脸左右对称，是安全的数据增强）

        x = torch.from_numpy(np.ascontiguousarray(img)).float()
        x = x.div_(127.5).sub_(1.0)  # [0,255] -> [-1,1]
        return x, index


def build_loader(
    root: str | Path,
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment: bool = False,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    cache_ram: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, CelebASplit]:
    """构造 DataLoader，shuffle/drop_last 默认按 split 自动决定。

    batch 形如 (images[ B,3,S,S] in [-1,1], index[B])。
    """
    is_train = split == "train"
    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = is_train

    ds = CelebASplit(
        root=root,
        split=split,
        augment=augment and is_train,
        cache_ram=cache_ram,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, ds


def read_meta(root: str | Path) -> dict:
    import json

    p = Path(root) / "meta.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
