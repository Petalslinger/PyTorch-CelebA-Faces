# 服务器端运行包

这个目录是**自包含**的：把整个 `server/` 上传到服务器，装好依赖就能直接跑，
不需要 git，也不需要项目里的其他文件（`archive.zip` 和 `data/` 单独传）。

> ⚠️ 这个目录由项目根目录的 `pack_server.sh` 生成，**不要直接改里面的代码**。
> 要改代码请改项目根目录的源文件，然后重新执行 `bash pack_server.sh`。

---

## 1. 上传到服务器

在**本地**项目根目录执行：

```bash
# 方式一：直接传目录（改动小的时候方便）
scp -r server user@server:~/celeba

# 方式二：先打包再传（文件多、网络慢时更快）
bash pack_server.sh                      # 生成 server_bundle.tar.gz
scp server_bundle.tar.gz user@server:~/
ssh user@server 'tar -xzf server_bundle.tar.gz'
```

服务器上最终结构：

```
~/celeba/
├── README.md            ← 就是这份文档
├── VERSION              ← 打包时的 commit 与时间
├── smoke_test.sh        ← 冒烟测试
├── run_all.sh           ← 一键全流程
├── requirements.txt
├── prepare_data.py  dataset.py  utils.py  metrics.py  fid.py
├── train_gan.py  train_vae.py  evaluate.py
└── models/
    ├── __init__.py  gan.py  vae.py
```

## 2. 装环境

```bash
conda create -n celeba python=3.10 -y
conda activate celeba

# 按服务器实际的 CUDA 版本选一个（下面以 CUDA 12.1 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 必须是 `True`，否则正式训练会慢到不可接受。

## 3. 准备数据（三选一）

### 方案 A：传 zip，在服务器上转换（推荐，传输量最小）

```bash
# 本地
scp archive.zip user@server:~/celeba/

# 服务器
cd ~/celeba
python prepare_data.py --zip archive.zip --out data/celeba64 --size 64
```

全量 20 万张，16 进程大约 5~15 分钟，产出 2.5 GB 的 npy。

### 方案 B：传本地已转换好的 npy（省 CPU，但传输量大）

```bash
# 本地执行（data/celeba64 下三个 npy 合计约 2.5 GB）
scp -r data/celeba64 user@server:~/celeba/data/
```

适合服务器 CPU 弱、或者本地已经转换过不想再等的情况。zip 是 1.4 GB，
转换后 2.5 GB，按网络情况选。

### 方案 C：服务器上已经有解压好的图片目录

```bash
python prepare_data.py --src /path/to/img_align_celeba --out data/celeba64 --size 64
```

### 转换完先看预览图

```bash
ls -lh data/celeba64/
# 把 preview.png 拉回本地看一眼，确认人脸是正的、没有黑边
```

```bash
# 本地执行
scp user@server:~/celeba/data/celeba64/preview.png .
```

## 4. 先冒烟测试

```bash
bash smoke_test.sh
```

几分钟内依次检查：CUDA 是否可用、数据能否读取、DataLoader 多进程是否被限制、
GAN 与 VAE 能否前向反向并存检查点、评估能否产出全部图。

看到「冒烟测试通过」再往下走。产物在 `_smoke/`，看完 `rm -rf _smoke`。

可调参数（环境变量）：

```bash
N=5000 STEPS=200 bash smoke_test.sh    # 多看一点
SIZE=128 bash smoke_test.sh            # 顺便验证 128 分辨率
WORKERS=4 bash smoke_test.sh           # 强制多进程
PY=/opt/conda/envs/celeba/bin/python bash smoke_test.sh
```

## 5. 正式训练

训练要跑几十分钟到几小时，**必须放在 tmux / screen / nohup 里**，
否则 SSH 一断进程就被杀。

```bash
tmux new -s celeba
conda activate celeba
cd ~/celeba
FULL=1 bash run_all.sh 2>&1 | tee train.log
# 按 Ctrl+B 再按 D 退出，训练继续在后台跑
# 回来查看：tmux attach -t celeba
```

不用 tmux 的话：

```bash
nohup bash run_all.sh > train.log 2>&1 &
tail -f train.log        # Ctrl+C 只是停止查看，不影响训练
```

`run_all.sh` 会依次跑：数据准备 → DCGAN → ConvVAE → 评估。

常用组合：

```bash
bash run_all.sh                          # 5k 子集 + 20 epoch，快速看效果
FULL=1 bash run_all.sh                   # 全量数据 + 30 epoch，正式结果
EPOCHS=50 BATCH=128 FULL=1 bash run_all.sh
```

### 想分开跑、中途调参

```bash
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --amp --save-every 5

python train_vae.py --data data/celeba64 --out runs/vae64 \
    --epochs 30 --batch-size 128 --lr 1e-3 --kl-warmup 5 --amp --save-every 5
```

`--save-every 5` 很重要：每个检查点约 70~90 MB（含优化器状态和 EMA），
默认每轮都存的话 30 轮就是 2~3 GB，很容易把磁盘塞满。

### 中断了怎么办

```bash
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --amp --resume runs/gan64/last.pt
```

`--resume` 会恢复模型、优化器、学习率调度器、EMA 和当前 epoch。

## 6. 评估

`run_all.sh` 最后一步会自动跑。单独跑：

```bash
python evaluate.py --data data/celeba64 --out runs/eval \
    --gan runs/gan64 --vae runs/vae64 --split test \
    --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
```

`--fid` 会额外下载 ImageNet 预训练的 Inception-v3（首次约 100 MB），
不需要可以去掉。

## 7. 取回结果

```bash
# 本地执行
scp -r user@server:~/celeba/runs/eval ./runs_eval
```

实验报告真正要用的文件：

| 文件 | 用途 |
| --- | --- |
| `report.md` | 自动生成的中文结论草稿，数字来自真实计算 |
| `results.json` | 全部逐图指标，写报告时查具体数值 |
| `compare_vae_recon.png` | 上排 GT / 下排 VAE 重建 → 定性评估 |
| `compare_gan_inversion.png` | 上排 GT / 下排 GAN 反演 → 定性评估 |
| `compare_gan_nn.png` | GAN 随机生成与最近邻真实图 |
| `samples_gan.png`、`samples_vae_prior.png` | 随机采样 |
| `metric_bar.png` | PSNR/SSIM 柱状图 |

训练过程图（`runs/gan64/samples/`、`runs/vae64/` 下的 `recon_*.png`）
也能用来说明训练是否收敛，需要的话一起拉回去。

## 8. 常见问题

**CUDA out of memory**
降 batch：`--batch-size 64` 或 `32`。64x64 分辨率下 128 batch 大约占 2~3 GB，
8 GB 的卡很宽裕；真爆了先看是不是有别的进程占着显存。

**`DataLoader worker ... PermissionError` / 卡在启动**
某些容器禁止多进程。`smoke_test.sh` 会自动探测并降级；手动跑时加 `--num-workers 0`。

**训练很慢，GPU 利用率低**
`--num-workers 0` 时数据加载会成为瓶颈。先跑 `nvidia-smi` 看 GPU 利用率，
低于 50% 就调大 `--num-workers 4~8`；确认磁盘是本地盘而不是网络盘。

**磁盘不够**
`--save-every 5` 减少检查点；`data/celeba64/*.npy` 用完可以删掉重新生成；
`runs/*/epoch_*.pt` 只保留最后几个。

**生成结果全是噪声**
先确认训练轮数够不够（GAN 至少 20~30 epoch 才有形状），
再看 `runs/gan64/loss.csv` 里 D 和 G 的损失是否还在正常范围。

**指标和 README 里的预期差很多**
README 第 5 节的区间是 30 epoch 量级的经验值，实际受数据量、超参影响。
以 `results.json` 为准，并在报告里说明训练配置。

---

## 附：这个包里有什么

| 文件 | 作用 |
| --- | --- |
| `prepare_data.py` | zip/文件夹 → train/val/test.npy（官方划分） |
| `dataset.py` | Dataset / DataLoader 封装 |
| `utils.py` | seed、device、EMA、checkpoint、日志 |
| `metrics.py` | PSNR / SSIM（纯 torch）、最近邻配对 |
| `fid.py` | FID（Inception-v3 特征） |
| `models/gan.py` | DCGAN 生成器与判别器 |
| `models/vae.py` | 卷积 VAE |
| `train_gan.py` | GAN 训练 |
| `train_vae.py` | VAE 训练 |
| `evaluate.py` | 定量 + 定性评估，自动生成结论 |
| `smoke_test.sh` | 冒烟测试 |
| `run_all.sh` | 一键全流程 |
