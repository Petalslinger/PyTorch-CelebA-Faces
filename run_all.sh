#!/usr/bin/env bash
# 一键跑完整流程：数据准备 -> 训练 GAN/VAE -> 评估
#
# 用法：
#   bash run_all.sh                              # 默认 5k 子集 + 20 epoch（快速验证）
#   FULL=1 bash run_all.sh                       # 全量数据 + 30 epoch
#   EPOCHS=50 BATCH=128 bash run_all.sh
#   WORKERS=0 bash run_all.sh                    # 容器禁止多进程时
#
# 数据来源（三者互斥，按顺序自动探测）：
#   ROOT=/root/autodl-pub/CelebA bash run_all.sh   # 官方目录（AutoDL 公共数据集）
#   ZIP=archive.zip bash run_all.sh                # Kaggle 压缩包
#   SRC=./img_align_celeba bash run_all.sh         # 已解压的图片目录
set -euo pipefail

PY=${PY:-python}
export PYTHONIOENCODING=utf-8
SIZE=${SIZE:-64}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-20}
WORKERS=${WORKERS:-4}
DATA=${DATA:-data/celeba${SIZE}}
GAN_OUT=${GAN_OUT:-runs/gan${SIZE}}
VAE_OUT=${VAE_OUT:-runs/vae${SIZE}}
EVAL_OUT=${EVAL_OUT:-runs/eval${SIZE}}

if [[ "${FULL:-0}" == "1" ]]; then
  PREP_EXTRA=""
  EPOCHS=${EPOCHS:-30}
else
  PREP_EXTRA="--max-images 5000"
fi

echo "== 1/4 数据准备 =="
if [[ -f "${DATA}/train.npy" ]]; then
  echo "   已存在 ${DATA}/train.npy，跳过"
else
  # 按 ROOT > ZIP > SRC > 自动探测 的顺序决定数据来源
  if [[ -n "${ROOT:-}" ]]; then
    SRC_ARG=(--root "${ROOT}")
  elif [[ -n "${ZIP:-}" ]]; then
    SRC_ARG=(--zip "${ZIP}")
  elif [[ -n "${SRC:-}" ]]; then
    SRC_ARG=(--src "${SRC}")
  elif [[ -d /root/autodl-pub/CelebA ]]; then
    echo "   自动使用 AutoDL 公共数据集 /root/autodl-pub/CelebA"
    SRC_ARG=(--root /root/autodl-pub/CelebA)
  elif [[ -f archive.zip ]]; then
    SRC_ARG=(--zip archive.zip)
  else
    cat >&2 <<'EOF'
   找不到数据来源。请用下面任一方式指定：
     ROOT=/root/autodl-pub/CelebA bash run_all.sh    # 官方目录结构
     ZIP=archive.zip bash run_all.sh                 # Kaggle 压缩包
     SRC=./img_align_celeba bash run_all.sh          # 已解压的图片目录
EOF
    exit 1
  fi
  $PY prepare_data.py "${SRC_ARG[@]}" --out "${DATA}" --size "${SIZE}" ${PREP_EXTRA}
fi

echo "== 2/4 训练 DCGAN =="
$PY train_gan.py --data "${DATA}" --out "${GAN_OUT}" \
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers "${WORKERS}" --amp

echo "== 3/4 训练 ConvVAE =="
$PY train_vae.py --data "${DATA}" --out "${VAE_OUT}" \
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --num-workers "${WORKERS}" \
  --lr 1e-3 --amp --kl-warmup 5

echo "== 4/4 评估 =="
$PY evaluate.py --data "${DATA}" --out "${EVAL_OUT}" \
  --gan "${GAN_OUT}" --vae "${VAE_OUT}" --num-workers "${WORKERS}" \
  --n-samples 1024 --inversion-n 32 --inversion-steps 200

echo "完成：结果见 ${EVAL_OUT}/report.md 与 ${EVAL_OUT}/compare_*.png"
