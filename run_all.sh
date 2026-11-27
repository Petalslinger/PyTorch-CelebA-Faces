#!/usr/bin/env bash
# 一键跑完整流程：数据准备 -> 训练 GAN/VAE -> 评估
# 用法：
#   bash run_all.sh                 # 默认 5k 子集 + 20 epoch（快速验证）
#   FULL=1 bash run_all.sh          # 全量数据 + 30 epoch
#   EPOCHS=50 BATCH=128 bash run_all.sh
set -euo pipefail

PY=${PY:-python}
SIZE=${SIZE:-64}
BATCH=${BATCH:-128}
EPOCHS=${EPOCHS:-20}
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
elif [[ -f archive.zip ]]; then
  $PY prepare_data.py --zip archive.zip --out "${DATA}" --size "${SIZE}" ${PREP_EXTRA}
else
  echo "   找不到 archive.zip，请先下载 Kaggle CelebA 数据集放到项目根目录" >&2
  exit 1
fi

echo "== 2/4 训练 DCGAN =="
$PY train_gan.py --data "${DATA}" --out "${GAN_OUT}" \
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --amp

echo "== 3/4 训练 ConvVAE =="
$PY train_vae.py --data "${DATA}" --out "${VAE_OUT}" \
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr 1e-3 --amp --kl-warmup 5

echo "== 4/4 评估 =="
$PY evaluate.py --data "${DATA}" --out "${EVAL_OUT}" \
  --gan "${GAN_OUT}" --vae "${VAE_OUT}" \
  --n-samples 1024 --inversion-n 32 --inversion-steps 200

echo "完成：结果见 ${EVAL_OUT}/report.md 与 ${EVAL_OUT}/compare_*.png"
