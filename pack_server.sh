#!/usr/bin/env bash
# 把服务器要用的文件收集到 server/ 目录（自包含，可直接 scp 上去跑）
#
#   bash pack_server.sh          # 生成 server/
#   bash pack_server.sh --tar    # 额外打包成 server_bundle.tar.gz
#
# 代码以项目根的源文件为准，server/ 里的副本请勿手改。
set -euo pipefail
cd "$(dirname "$0")"

DEST=server
FILES=(
  prepare_data.py
  dataset.py
  utils.py
  metrics.py
  fid.py
  train_gan.py
  train_vae.py
  evaluate.py
  smoke_test.sh
  run_all.sh
  requirements.txt
)
MODELS=(models/__init__.py models/gan.py models/vae.py)

say() { printf '\033[1;36m== %s ==\033[0m\n' "$*"; }

say "清理旧的 ${DEST}/"
rm -rf "$DEST"
mkdir -p "$DEST/models"

say "复制代码"
for f in "${FILES[@]}" "${MODELS[@]}"; do
  [[ -f "$f" ]] || { echo "缺少源文件: $f" >&2; exit 1; }
  cp "$f" "$DEST/$f"
  printf '   %s\n' "$f"
done

say "复制部署文档"
cp SERVER_DEPLOY.md "$DEST/README.md"
printf '   SERVER_DEPLOY.md -> %s/README.md\n' "$DEST"

say "写入 VERSION"
{
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
  echo "branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
  echo "packed: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "source: $(pwd)"
} > "$DEST/VERSION"
cat "$DEST/VERSION" | sed 's/^/   /'

say "语法检查"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
if command -v "$PY" >/dev/null 2>&1; then
  ( cd "$DEST" && "$PY" - <<'PYEOF'
import pathlib, py_compile, sys
bad = 0
for p in sorted(pathlib.Path(".").rglob("*.py")):
    try:
        py_compile.compile(str(p), doraise=True, cfile=str(p) + "c")
    except Exception as e:
        print(f"   ✗ {p}: {e}")
        bad = 1
    finally:
        pathlib.Path(str(p) + "c").unlink(missing_ok=True)
if bad:
    sys.exit(1)
print("   所有 .py 编译通过")
PYEOF
  ) || { echo "语法检查失败" >&2; exit 1; }
  bash -n "$DEST/smoke_test.sh" && echo "   smoke_test.sh 语法通过"
  bash -n "$DEST/run_all.sh"   && echo "   run_all.sh 语法通过"
else
  echo "   跳过（找不到 python: $PY）"
fi

say "完成"
du -sh "$DEST" | sed 's/^/   /'
find "$DEST" -type f | sort | sed 's/^/   /'

if [[ "${1:-}" == "--tar" ]]; then
  say "打包"
  tar -czf server_bundle.tar.gz -C . "$DEST"
  ls -lh server_bundle.tar.gz | awk '{print "   server_bundle.tar.gz  " $5}'
  echo
  echo "   上传：scp server_bundle.tar.gz user@server:~/"
  echo "   解压：ssh user@server 'tar -xzf server_bundle.tar.gz'"
else
  echo
  echo "   上传：scp -r ${DEST} user@server:~/celeba"
  echo "   或先打包：bash pack_server.sh --tar"
fi
