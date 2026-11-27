#!/usr/bin/env bash
# 收集训练与评估产物，打包成一个便于下载的压缩包。
#
#   bash collect_results.sh                        # 自动探测 runs/ 下的 gan/vae/eval
#   bash collect_results.sh --with-ckpt            # 附带 last.pt / best.pt
#   bash collect_results.sh --gan runs/gan64 --vae runs/vae64 --eval runs/eval
#   bash collect_results.sh --out 我的结果          # 指定包名
#   bash collect_results.sh --help                 # 全部选项
#
# 产物结构：
#   results_<时间戳>/
#   ├── README.txt      结果说明与指标口径
#   ├── MANIFEST.txt    文件清单（大小 + md5）
#   ├── env.txt         运行环境（GPU / torch / python）
#   ├── VERSION         代码版本
#   ├── eval/           report.md、results.json、全部对比图
#   ├── gan/            loss.csv、train_summary.json、samples/
#   └── vae/            同上
set -euo pipefail
cd "$(dirname "$0")"

GAN_DIR=""
VAE_DIR=""
EVAL_DIR=""
DATA_DIR=""
OUT_NAME=""
RUNS_DIR=""
WITH_CKPT=0
WITH_EPOCHS=0
WITH_DATA=0
KEEP=0

say()  { printf '\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31m   %s\033[0m\n' "$*" >&2; }

usage() {
  cat <<'USAGE'
收集训练与评估产物，打包成一个便于下载的压缩包。

用法：
  bash collect_results.sh                        # 自动探测 runs/ 下的 gan/vae/eval
  bash collect_results.sh --with-ckpt            # 附带 last.pt / best.pt
  bash collect_results.sh --with-epochs          # 附带全部 epoch_*.pt
  bash collect_results.sh --gan runs/gan64 --vae runs/vae64 --eval runs/eval
  bash collect_results.sh --out 我的结果          # 指定包名（不含路径）

选项：
  --gan DIR          GAN 训练目录（含 hparams.json / loss.csv）
  --vae DIR          VAE 训练目录
  --eval DIR         评估目录（含 results.json / report.md）
  --runs DIR         结果根目录，在此下面自动查找（默认依次尝试
                     runs/、../runs、/root/autodl-tmp/runs 等）
  --data DIR         附带数据集的 meta.json 与 preview.png
  --out NAME         输出包名，默认 results_<时间戳>
  --with-ckpt        打包 last.pt 与 best.pt（每个 70~90 MB）
  --with-epochs      打包全部 epoch_*.pt（体积很大）
  --keep             打包后保留解压出来的暂存目录
  -h, --help         显示本帮助

产物结构：
  results_<时间戳>/
  ├── README.txt      结果说明与指标口径
  ├── MANIFEST.txt    文件清单（大小 + md5）
  ├── env.txt         运行环境（GPU / torch / python）
  ├── VERSION         代码版本
  ├── eval/           report.md、results.json、全部对比图
  ├── gan/            loss.csv、train_summary.json、samples/
  └── vae/            同上
USAGE
  exit 0
}

need_val() { [[ $# -ge 2 ]] || { err "$1 后面缺少目录/名称"; exit 1; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gan)         need_val "$@"; GAN_DIR="$2";  shift 2 ;;
    --vae)         need_val "$@"; VAE_DIR="$2";  shift 2 ;;
    --eval)        need_val "$@"; EVAL_DIR="$2"; shift 2 ;;
    --data)        need_val "$@"; DATA_DIR="$2"; WITH_DATA=1; shift 2 ;;
    --runs)        need_val "$@"; RUNS_DIR="$2"; shift 2 ;;
    --out)         need_val "$@"; OUT_NAME="$2"; shift 2 ;;
    --with-ckpt)   WITH_CKPT=1; shift ;;
    --with-epochs) WITH_CKPT=1; WITH_EPOCHS=1; shift ;;
    --keep)        KEEP=1; shift ;;
    -h|--help)     usage ;;
    *) err "未知参数: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- 自动探测
# 在候选目录里找“看起来像”的那一个：GAN 的 hparams.json 里 model=dcgan，
# VAE 的是 model=convvae，评估目录有 results.json。
# 注意 json 由 save_json(indent=2) 写出，冒号后有一个空格。
looks_like_gan()  { [[ -f "$1/hparams.json" ]] && grep -q '"model": *"dcgan"'   "$1/hparams.json" 2>/dev/null; }
looks_like_vae()  { [[ -f "$1/hparams.json" ]] && grep -q '"model": *"convvae"' "$1/hparams.json" 2>/dev/null; }
looks_like_eval() { [[ -f "$1/results.json" ]]; }

# 候选根目录：脚本所在目录下的 runs/，以及 AutoDL 常见的 /root/autodl-tmp 等。
# 训练产物通常不在代码目录里（数据盘挂载在别处），所以必须往外找。
runs_roots() {
  local r
  [[ -n "$RUNS_DIR" ]] && printf '%s\n' "$RUNS_DIR"
  for r in runs ../runs ../../runs \
           autodl-tmp/runs ../autodl-tmp/runs ../../autodl-tmp/runs \
           "${HOME:-/root}/autodl-tmp/runs" /root/autodl-tmp/runs \
           autodl-tmp ../autodl-tmp "${HOME:-/root}/autodl-tmp" /root/autodl-tmp \
           output out results; do
    [[ -d "$r" ]] && printf '%s\n' "$r"
  done
  return 0
}

# 先试候选根目录本身，再试它下面一层子目录
autodetect() {  # $1 = 判断函数名
  local fn="$1"
  local -a cands=()
  local base sub
  while IFS= read -r base; do
    [[ -d "$base" ]] || continue
    cands+=("$base")
    for sub in "$base"/*/; do
      [[ -d "$sub" ]] && cands+=("${sub%/}")
    done
  done < <(runs_roots)
  local d
  for d in ${cands[@]+"${cands[@]}"}; do
    [[ -d "$d" ]] || continue
    if "$fn" "$d"; then printf '%s\n' "$d"; return 0; fi
  done
  return 1
}

# ---------------------------------------------------------------- 校验
say "检查输入目录"
found_any=0

check_dir() {  # $1=标签 $2=路径 $3=必需文件
  [[ -n "$2" ]] || return 0
  if [[ ! -d "$2" ]]; then
    err "$1 目录不存在: $2"; exit 1
  fi
  if [[ -n "$3" && ! -f "$2/$3" ]]; then
    warn "$1 目录里没有 $3，可能没训练完: $2"
  fi
  found_any=1
}

if [[ -z "$GAN_DIR" ]]; then  GAN_DIR="$(autodetect looks_like_gan  2>/dev/null || true)"; fi
if [[ -z "$VAE_DIR" ]]; then  VAE_DIR="$(autodetect looks_like_vae  2>/dev/null || true)"; fi
if [[ -z "$EVAL_DIR" ]]; then EVAL_DIR="$(autodetect looks_like_eval 2>/dev/null || true)"; fi

check_dir GAN  "$GAN_DIR"  "loss.csv"
check_dir VAE  "$VAE_DIR"  "loss.csv"
check_dir EVAL "$EVAL_DIR" "results.json"

if [[ "$found_any" -eq 0 ]]; then
  err "没有找到任何结果目录。请用 --runs / --gan / --vae / --eval 指定，例如："
  err "  bash collect_results.sh --runs /root/autodl-tmp/runs"
  echo >&2
  echo "   当前目录：$(pwd)" >&2
  echo "   查找过的路径（只列存在的）：" >&2
  local_roots="$(runs_roots 2>/dev/null)"
  if [[ -n "$local_roots" ]]; then
    printf '%s\n' "$local_roots" | sed 's/^/     /' >&2
  else
    echo "     （无）" >&2
  fi
  echo >&2
  echo "   若训练产物在其他路径，用 --runs 指定其父目录，或直接指定三类目录：" >&2
  echo "     --gan <GAN输出目录> --vae <VAE输出目录> --eval <评估输出目录>" >&2
  exit 1
fi

[[ -n "$GAN_DIR"  ]] && echo "   GAN:  $GAN_DIR"
[[ -n "$VAE_DIR"  ]] && echo "   VAE:  $VAE_DIR"
[[ -n "$EVAL_DIR" ]] && echo "   EVAL: $EVAL_DIR"

# ---------------------------------------------------------------- 准备目录
TS="$(date '+%Y%m%d_%H%M%S')"
[[ -n "$OUT_NAME" ]] || OUT_NAME="results_${TS}"
# 包名只保留安全字符，避免空格/斜杠导致 tar 失败
OUT_NAME="$(printf '%s' "$OUT_NAME" | tr ' /\\:' '____')"

# 暂存目录用 mktemp，绝不用用户给的名字——否则 --out runs 会把真实结果目录删掉
WORK="$(mktemp -d "${TMPDIR:-/tmp}/collect_results.XXXXXX")"
cleanup() { [[ "$KEEP" -eq 1 ]] && return 0; rm -rf "$WORK"; }
trap cleanup EXIT INT TERM
STAGE="${WORK}/${OUT_NAME}"
mkdir -p "$STAGE"

say "收集文件"
copied=0
copy_if() {  # $1=源 $2=目标
  if [[ -f "$1" ]]; then
    mkdir -p "$(dirname "$2")"
    cp "$1" "$2"
    copied=$((copied + 1))
  fi
}

# ---- 评估结果：报告要用的核心文件 ----
if [[ -n "$EVAL_DIR" ]]; then
  for f in results.json report.md hparams.json; do
    copy_if "$EVAL_DIR/$f" "$STAGE/eval/$f"
  done
  for f in "$EVAL_DIR"/*.png; do
    [[ -f "$f" ]] && copy_if "$f" "$STAGE/eval/$(basename "$f")"
  done
  echo "   eval/    $(find "$STAGE/eval" -type f 2>/dev/null | wc -l | tr -d ' ') 个文件"
fi

# ---- 训练过程：日志、摘要、每轮样本图 ----
collect_train() {  # $1=源目录 $2=目标子目录
  local src="$1" dst="$2"
  [[ -n "$src" ]] || return 0
  for f in loss.csv train_summary.json hparams.json; do
    copy_if "$src/$f" "$STAGE/$dst/$f"
  done
  local n=0
  if [[ -d "$src/samples" ]]; then
    for f in "$src/samples"/*.png; do
      [[ -f "$f" ]] || continue
      copy_if "$f" "$STAGE/$dst/samples/$(basename "$f")"
      n=$((n + 1))
    done
  fi
  echo "   $dst/     $(find "$STAGE/$dst" -type f 2>/dev/null | wc -l | tr -d ' ') 个文件（含 $n 张样本图）"
}

collect_train "$GAN_DIR" "gan"
collect_train "$VAE_DIR" "vae"

# ---- 检查点（可选）----
collect_ckpt() {  # $1=源目录 $2=目标子目录
  local src="$1" dst="$2"
  [[ -n "$src" ]] || return 0
  for f in last.pt best.pt; do
    copy_if "$src/$f" "$STAGE/$dst/$f"
  done
  if [[ "$WITH_EPOCHS" -eq 1 ]]; then
    for f in "$src"/epoch_*.pt; do
      [[ -f "$f" ]] && copy_if "$f" "$STAGE/$dst/$(basename "$f")"
    done
  fi
}

if [[ "$WITH_CKPT" -eq 1 ]]; then
  collect_ckpt "$GAN_DIR" "gan"
  collect_ckpt "$VAE_DIR" "vae"
  echo "   检查点  $(find "$STAGE" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ') 个"
fi

# ---- 数据元信息（很小，但能说明用了多少数据）----
# 未显式指定时，跟随数据目录自动找（meta.json 里记着图片数与划分来源）。
# 数据通常放在 <root>/data/celeba64/ 这样的两级结构里，故也查下一层。
if [[ "$WITH_DATA" -eq 0 ]]; then
  for d in data ../data ../../data autodl-tmp/data ../autodl-tmp/data \
           "${HOME:-/root}/autodl-tmp/data" /root/autodl-tmp/data; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/meta.json" ]]; then DATA_DIR="$d"; WITH_DATA=1; break; fi
    hit="$(find "$d" -maxdepth 2 -name meta.json -print -quit 2>/dev/null || true)"
    if [[ -n "$hit" ]]; then
      DATA_DIR="$(dirname "$hit")"; WITH_DATA=1; break
    fi
  done
fi
if [[ "$WITH_DATA" -eq 1 ]]; then
  if [[ -d "$DATA_DIR" ]]; then
    for f in meta.json preview.png; do
      copy_if "$DATA_DIR/$f" "$STAGE/data/$f"
    done
    echo "   data/    meta.json + preview.png（来自 $DATA_DIR）"
  else
    warn "--data 指定的目录不存在: $DATA_DIR"
  fi
fi

# ---------------------------------------------------------------- 环境信息
say "写入说明文件"

{
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
  echo "branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
  echo "packed: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "host:   $(hostname 2>/dev/null || echo 'unknown')"
} > "$STAGE/VERSION"

# 找一个能 import torch 的解释器来记录环境（与 smoke_test.sh 同思路，取第一个能用的）
find_torch_python() {
  local -a cands=()
  [[ -n "${PY:-}" ]] && cands+=("$PY")
  cands+=(
    "${CONDA_PREFIX:-}/bin/python"
    /root/miniconda3/envs/celeba/bin/python
    /root/miniconda3/bin/python
    /opt/conda/bin/python
    "${HOME:-/root}/miniconda3/bin/python"
    "${HOME:-/root}/anaconda3/bin/python"
    python3 python
  )
  local base
  for base in "${CONDA_PREFIX:-}" /root/miniconda3 /opt/conda \
              "${HOME:-/root}/miniconda3" "${HOME:-/root}/anaconda3"; do
    [[ -d "$base/envs" ]] || continue
    local e
    for e in "$base"/envs/*/bin/python "$base"/envs/*/python.exe; do
      [[ -x "$e" ]] && cands+=("$e")
    done
  done
  local c
  # 两轮：先找带 CUDA 的（真正训练用的那个），没有再退而求其次
  for want in cuda any; do
    for c in "${cands[@]}"; do
      [[ -n "$c" ]] || continue
      command -v "$c" >/dev/null 2>&1 || continue
      if [[ "$want" == "cuda" ]]; then
        "$c" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1 \
          && { printf '%s\n' "$c"; return 0; }
      else
        "$c" -c 'import torch' >/dev/null 2>&1 \
          && { printf '%s\n' "$c"; return 0; }
      fi
    done
  done
  return 1
}

PY_BIN="$(find_torch_python 2>/dev/null || true)"
{
  if [[ -n "$PY_BIN" ]]; then
    echo "python: $("$PY_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
    echo "torch:  $("$PY_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
    echo "cuda:   $("$PY_BIN" -c 'import torch; print(torch.version.cuda or "none")' 2>/dev/null || echo unknown)"
    echo "gpu:    $("$PY_BIN" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null || echo unknown)"
    echo "interp: $PY_BIN"
  else
    echo "python: unknown（未找到能 import torch 的解释器）"
  fi
} > "$STAGE/env.txt"
sed 's/^/   /' "$STAGE/env.txt"

cat > "$STAGE/README.txt" <<'TXTEOF'
实验结果包说明
==============

目录结构
--------
eval/    评估产物，写报告主要用这里
  report.md                    自动生成的中文结论草稿，数值来自真实计算
  results.json                 全部逐图指标 + 完整运行配置
  compare_vae_recon.png        上排 GT / 下排 VAE 重建      ← 定性评估用
  compare_gan_inversion.png    上排 GT / 下排 GAN 反演重建  ← 定性评估用
  compare_gan_nn.png           GAN 随机生成 vs 最近邻真实图
  samples_gan.png              GAN 随机采样网格
  samples_vae_prior.png        VAE 先验采样网格
  samples_vae_recon.png        VAE 重建网格
  samples_gan_inversion.png    GAN 反演重建网格
  metric_bar.png               PSNR / SSIM 柱状图

gan/     训练过程（GAN）
vae/     训练过程（VAE）
  loss.csv                    每轮损失，可用 Excel 直接画曲线
  train_summary.json          最佳轮次、总耗时
  hparams.json                完整超参数，复现实验时需要
  samples/                    每轮生成图，用来说明是否收敛

VERSION  打包时的代码版本
env.txt  运行环境（GPU 型号、torch 版本）
MANIFEST.txt  文件清单与 md5 校验值

指标口径（报告中必须说明）
--------------------------
1. VAE-recon
   输入真实图 x，比较重建结果 x̂ 与 x。GT 就是 x 本身，一一对应，口径最标准。

2. GAN-inversion
   为每张真实图优化一个潜变量 z，使 G(z) ≈ x，得到与 VAE 同口径的重建指标。
   反映生成器能覆盖多少真实图像。

3. GAN-NN / VAE-prior-NN
   GAN 与 VAE 先验采样都是无条件生成，输出与任何真实图都没有对应关系。
   这里为每张生成图在测试集里检索像素空间最相近的真实图作为“伪 GT”。
   该口径只能说明生成分布是否接近真实数据，不能作为重建质量指标，
   也不衡量图像清晰度，必须配合 FID 与人眼判断。

4. FID
   分布级指标，越低越好。需要不少于 1000 张样本，依赖 ImageNet 预训练的 Inception-v3。
   未计算时 results.json 中不会有 fid 字段。

注意事项
--------
* PSNR / SSIM 是逐像素指标，对 GAN 天然不利：GAN 的对抗损失不优化像素对齐，
  它的优势在感知质量（纹理锐利、结构自然），要用定性图说明。
* 所有数值均来自真实计算，未做任何手工调整。
* report.md 是自动生成的模板结论，写入实验报告前应结合对比图人工复核。
TXTEOF
echo "   README.txt"

# ---------------------------------------------------------------- 清单
# 清单本身不列入清单，避免自引用
HAS_MD5=0
command -v md5sum >/dev/null 2>&1 && HAS_MD5=1

manifest() (
  cd "$STAGE"
  if [[ "$HAS_MD5" -eq 1 ]]; then
    printf '%-50s %8s  %s\n' "文件" "大小" "md5"
    printf '%.0s-' {1..80}; printf '\n'
  else
    printf '%-50s %8s\n' "文件" "大小"
    printf '%.0s-' {1..64}; printf '\n'
  fi
  local f
  while IFS= read -r f; do
    f="${f#./}"
    if [[ "$HAS_MD5" -eq 1 ]]; then
      printf '%-50s %8s  %s\n' "$f" "$(du -h "$f" | cut -f1)" "$(md5sum "$f" | cut -d' ' -f1)"
    else
      printf '%-50s %8s\n' "$f" "$(du -h "$f" | cut -f1)"
    fi
  done < <(find . -type f ! -name MANIFEST.txt | sort)
)
manifest > "$STAGE/MANIFEST.txt"
echo "   MANIFEST.txt（$(wc -l < "$STAGE/MANIFEST.txt" | tr -d ' ') 行）"
[[ "$HAS_MD5" -eq 0 ]] && warn "未找到 md5sum，清单中不含校验值"

# ---------------------------------------------------------------- 打包
say "打包"
TARBALL="${OUT_NAME}.tar.gz"
tar -czf "$TARBALL" -C "$WORK" "$OUT_NAME"

SIZE="$(du -h "$TARBALL" | cut -f1)"
NFILES="$(tar -tzf "$TARBALL" | grep -vc '/$' || true)"
echo "   $TARBALL   $SIZE   $NFILES 个文件"

if [[ "$KEEP" -eq 1 ]]; then
  echo "   暂存目录保留在：$STAGE"
fi

echo
echo "   下载到本地："
echo "     scp user@server:$(pwd)/$TARBALL ./"
echo "   解压查看："
echo "     tar -xzf $TARBALL"

# ---------------------------------------------------------------- 提示
echo
if [[ "$WITH_CKPT" -eq 0 && ( -n "$GAN_DIR" || -n "$VAE_DIR" ) ]]; then
  echo "   提示：检查点未包含（体积大）。如需一并取回："
  echo "     bash collect_results.sh --with-ckpt        # last.pt + best.pt"
  echo "     bash collect_results.sh --with-epochs      # 外加全部 epoch_*.pt"
fi
if [[ -z "$EVAL_DIR" ]]; then
  echo "   提示：没有找到评估结果。先跑 evaluate.py，报告要用的图表都在那里。"
fi
