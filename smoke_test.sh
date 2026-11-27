#!/usr/bin/env bash
# 小规模冒烟测试：验证数据管线、两个模型、评估脚本都能在这台机器上跑通。
# 只跑几十步，不追求效果，几分钟内结束。
#
# 用法：
#   bash smoke_test.sh                    # 默认 2000 张 + 1 epoch + 50 step
#   N=5000 STEPS=200 bash smoke_test.sh   # 想多看一点
#   SIZE=128 bash smoke_test.sh           # 顺便验证 128 分辨率
#   WORKERS=0 bash smoke_test.sh          # 强制单进程（容器禁止多进程时）
#
# 数据来源（按顺序探测，用哪个都行）：
#   ROOT=/root/autodl-pub/CelebA bash smoke_test.sh   # AutoDL 公共数据集（官方目录）
#   DATA_SRC=./img_align_celeba bash smoke_test.sh    # 已解压的图片目录
#   ZIP=archive.zip bash smoke_test.sh                # Kaggle 压缩包
#   （都不指定时：先看已有 npy，再看 /root/autodl-pub/CelebA，最后找 archive.zip）
#
# 环境变量：
#   PY       python 解释器（默认 python）
#   SIZE     分辨率（默认 64）
#   N        取多少张图（默认 2000）
#   EPOCHS   训练轮数（默认 1）
#   STEPS    每轮步数上限（默认 50）
#   WORKERS  DataLoader 进程数（默认 2，探测失败自动降为 0）
#   OUT      所有产物目录（默认 _smoke，跑完可以直接删）
#   DATA     输出 npy 的目录（默认 ${OUT}/celeba${SIZE}）
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
export PYTHONIOENCODING=utf-8
SIZE=${SIZE:-64}
N=${N:-2000}
EPOCHS=${EPOCHS:-1}
STEPS=${STEPS:-50}
OUT=${OUT:-_smoke}
DATA=${DATA:-${OUT}/celeba${SIZE}}

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m   ✓ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m   ✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. 环境
say "0/6 环境检查"
$PY -c "import sys; print('   python', sys.version.split()[0], '->', sys.executable)"
$PY - <<'EOF' || die "torch/torchvision 不可用，先按 README 第 0 节安装"
import torch, torchvision
print(f"   torch {torch.__version__}  torchvision {torchvision.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"   cuda:{i}  {torch.cuda.get_device_name(i)}  "
              f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GiB")
else:
    print("   警告：CUDA 不可用，将用 CPU 跑（冒烟测试够用，正式训练会很慢）")
EOF
ok "依赖正常"

# DataLoader 多进程在容器/受限环境里可能被禁（Windows 上是 PermissionError），探测一下
WORKERS=${WORKERS:-2}
if [[ "${WORKERS}" -gt 0 ]]; then
  if $PY - <<'EOF' 2>/dev/null
import sys, torch
from torch.utils.data import DataLoader, TensorDataset
x = torch.zeros(4, 3, 8, 8)
try:
    next(iter(DataLoader(TensorDataset(x), batch_size=2, num_workers=1)))
except Exception:
    sys.exit(1)
EOF
  then
    ok "DataLoader 多进程可用（num_workers=${WORKERS}）"
  else
    WORKERS=0
    printf '\033[1;33m   注意：当前环境不允许 DataLoader 多进程，自动改用 num_workers=0\033[0m\n'
  fi
fi

# ---------------------------------------------------------------- 1. 数据
say "1/6 数据准备（${N} 张，${SIZE}x${SIZE}）"
PREP_HINT="--root /root/autodl-pub/CelebA"   # 末尾提示里展示的命令，下面会按实际情况覆盖
if [[ -f "${DATA}/train.npy" ]]; then
  ok "已存在 ${DATA}/train.npy，跳过"
elif [[ -f "data/celeba${SIZE}/train.npy" ]]; then
  # 之前已经全量转换过，直接复用，省时间
  DATA="data/celeba${SIZE}"
  ok "复用已准备好的数据 ${DATA}"
elif [[ -n "${ROOT:-}" ]]; then
  PREP_HINT="--root ${ROOT}"
  $PY prepare_data.py --root "${ROOT}" --out "${DATA}" --size "${SIZE}" --max-images "${N}"
elif [[ -n "${DATA_SRC:-}" && -d "${DATA_SRC}" ]]; then
  PREP_HINT="--src ${DATA_SRC}"
  $PY prepare_data.py --src "${DATA_SRC}" --out "${DATA}" --size "${SIZE}" --max-images "${N}"
elif [[ -d /root/autodl-pub/CelebA ]]; then
  echo "   自动使用 AutoDL 公共数据集 /root/autodl-pub/CelebA"
  PREP_HINT="--root /root/autodl-pub/CelebA"
  $PY prepare_data.py --root /root/autodl-pub/CelebA --out "${DATA}" --size "${SIZE}" --max-images "${N}"
elif [[ -n "${ZIP:-}" ]]; then
  echo "   使用压缩包 ${ZIP}"
  PREP_HINT="--zip ${ZIP}"
  $PY prepare_data.py --zip "${ZIP}" --out "${DATA}" --size "${SIZE}" --max-images "${N}"
else
  # 优先项目根的 archive.zip，其次当前目录，最后再往上找一层
  ZIP=""
  for c in archive.zip ../archive.zip ../../archive.zip; do
    [[ -f "$c" ]] && { ZIP="$c"; break; }
  done
  [[ -n "$ZIP" ]] || die "找不到数据。用 ROOT=<官方目录> / DATA_SRC=<图片目录> / ZIP=<压缩包> 指定其一"
  echo "   使用 ${ZIP}"
  PREP_HINT="--zip ${ZIP}"
  $PY prepare_data.py --zip "${ZIP}" --out "${DATA}" --size "${SIZE}" --max-images "${N}"
fi

$PY - "${DATA}" <<'EOF' || die "数据检查失败"
import sys, json
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
for split in ("train", "val", "test"):
    a = np.load(d / f"{split}.npy", mmap_mode="r")
    assert a.ndim == 4 and a.shape[1:] == (3, meta["image_size"], meta["image_size"]), a.shape
    assert a.dtype == np.uint8, a.dtype
    print(f"   {split:5s} {str(a.shape):22s} 像素范围 {a[:50].min()}~{a[:50].max()}")
print("   preview.png 存在" if (d / "preview.png").exists() else "   警告：没有 preview.png")
EOF
ok "数据就绪（建议打开 ${DATA}/preview.png 确认人脸方向正确）"

# 数据量太小的话 GAN 学不出东西，提一句
TRAIN_N=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['counts']['train'])" "${DATA}/meta.json")
if [[ "${TRAIN_N}" -lt 20000 ]]; then
  printf '\033[1;33m   注意：训练集只有 %s 张，冒烟测试可以，正式训练建议用全量 20 万张\033[0m\n' "${TRAIN_N}"
fi

# ---------------------------------------------------------------- 2. GAN
say "2/6 训练 DCGAN（${EPOCHS} epoch × ${STEPS} step）"
$PY train_gan.py --data "${DATA}" --out "${OUT}/gan" \
  --epochs "${EPOCHS}" --max-steps "${STEPS}" --batch-size 64 \
  --num-workers "${WORKERS}" --sample-n 16 --log-every 10 --amp
[[ -f "${OUT}/gan/last.pt" ]] || die "GAN 没有产出 last.pt"
ok "GAN 检查点已生成"

# ---------------------------------------------------------------- 3. VAE
say "3/6 训练 ConvVAE（${EPOCHS} epoch × ${STEPS} step）"
$PY train_vae.py --data "${DATA}" --out "${OUT}/vae" \
  --epochs "${EPOCHS}" --max-steps "${STEPS}" --batch-size 64 \
  --num-workers "${WORKERS}" --lr 1e-3 --sample-n 16 --log-every 10 --amp
[[ -f "${OUT}/vae/last.pt" ]] || die "VAE 没有产出 last.pt"
ok "VAE 检查点已生成"

# ---------------------------------------------------------------- 4. 评估
say "4/6 评估（小样本，只为验证流程）"
$PY evaluate.py --data "${DATA}" --out "${OUT}/eval" \
  --gan "${OUT}/gan" --vae "${OUT}/vae" --split test \
  --n-samples 64 --n-gallery 128 --inversion-n 4 --inversion-steps 30 \
  --batch-size 32 --num-workers "${WORKERS}"
ok "评估完成"

# ---------------------------------------------------------------- 5. 产物
say "5/6 检查产物"
need=(
  "${OUT}/gan/last.pt" "${OUT}/gan/loss.csv" "${OUT}/gan/hparams.json"
  "${OUT}/vae/last.pt" "${OUT}/vae/loss.csv" "${OUT}/vae/hparams.json"
  "${OUT}/eval/results.json" "${OUT}/eval/report.md"
  "${OUT}/eval/compare_vae_recon.png" "${OUT}/eval/compare_gan_inversion.png"
  "${OUT}/eval/samples_gan.png" "${OUT}/eval/samples_vae_prior.png"
)
miss=0
for f in "${need[@]}"; do
  if [[ -f "$f" ]]; then ok "$f"; else printf '\033[1;31m   ✗ 缺少 %s\033[0m\n' "$f"; miss=1; fi
done
[[ $miss -eq 0 ]] || die "有产物缺失"

$PY - "${OUT}/eval/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
summary = r.get("summary", {})
print("\n   本次冒烟测试的指标（步数极少，数值无参考意义）：")
for name, v in summary.items():
    ps, ss = v.get("psnr"), v.get("ssim")
    if ps is None or ss is None:
        continue
    fid = f"  FID {v['fid']:.1f}" if "fid" in v else ""
    print(f"   {name:16s} PSNR {ps['mean']:6.2f} dB   SSIM {ss['mean']:.4f}{fid}")
if not summary:
    raise SystemExit("   results.json 里没有 summary，评估可能没跑完整")
EOF

# ---------------------------------------------------------------- 6. 结论
say "6/6 全部通过"
cat <<EOF

  冒烟测试通过，说明数据、模型、训练、评估四条链路都能跑。
  注意：这里只跑了 ${EPOCHS} 个 epoch × ${STEPS} step，生成结果必然是噪声，
        results.json 里的数字没有参考价值。

  接下来正式训练：
    python prepare_data.py ${PREP_HINT} --out data/celeba${SIZE} --size ${SIZE}
    python train_gan.py --data data/celeba${SIZE} --out runs/gan${SIZE} --epochs 30 --batch-size 128 --amp
    python train_vae.py --data data/celeba${SIZE} --out runs/vae${SIZE} --epochs 30 --batch-size 128 --lr 1e-3 --kl-warmup 5 --amp
    python evaluate.py --data data/celeba${SIZE} --out runs/eval --gan runs/gan${SIZE} --vae runs/vae${SIZE} --n-samples 2048 --inversion-n 64 --fid

  或者直接：FULL=1 bash run_all.sh

  清理冒烟测试产物：rm -rf ${OUT}
EOF
