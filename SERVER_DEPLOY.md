# 服务器端运行包

这个目录是**自包含**的：把整个 `server/` 上传到服务器，装好依赖就能直接跑，
不需要 git，也不需要项目里的其他文件。

> **在 AutoDL 上**：如果公共数据集里已有 CelebA，那 `archive.zip` 和 `data/`
> 都不用传，`prepare_data.py --root /root/autodl-pub/CelebA` 直接从
> `/root/autodl-pub/CelebA/Img/img_align_celeba.zip` 读，见第 3 节方案 0。

> ⚠️ 这个目录由项目根目录的 `pack_server.sh` 生成，**不要直接改里面的代码**。
> 要改代码请改项目根目录的源文件，然后重新执行 `bash pack_server.sh`。

---

## 0. 完整命令（从零到出结果）

如果只想照着敲、不想读后面的说明，用这一节就够了。
假设是 **AutoDL**：conda 在 `/root/miniconda3`，公共数据集在
`/root/autodl-pub/CelebA`，输出写 `/root/autodl-tmp`（系统盘只有 30 G）。

### 步骤 1：本地打包上传

```bash
# 在本地项目根目录 PyTorch-CelebA-Faces/ 下执行
bash pack_server.sh --tar
scp server_bundle.tar.gz root@你的实例IP:~/
```

### 步骤 2：服务器上解压

```bash
ssh root@你的实例IP

cd ~
tar -xzf server_bundle.tar.gz
cd server
cat VERSION            # 确认 commit 是你刚打的包
```

### 步骤 3：确认 python 环境

AutoDL 镜像一般已经装好 torch，先看一眼：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 打印出 `True` 和显卡名字 → 跳过步骤 4，直接进步骤 5。
- 报 `No module named 'torch'` → 做步骤 4。

### 步骤 4：（仅当没有 torch 时）装环境

```bash
conda create -n celeba python=3.10 -y
conda activate celeba

# 按 nvidia-smi 右上角的 CUDA Version 选，下面以 12.1 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 步骤 5：转换数据（约 5~15 分钟）

先 dry-run 确认路径检测正确：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out /root/autodl-tmp/data/celeba64 --dry-run
```

输出里应该看到 `图片源: 压缩包 Img/img_align_celeba.zip（无需解压，直接读）`，
然后正式转换：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out /root/autodl-tmp/data/celeba64 --size 64 --workers 8
```

产出 `/root/autodl-tmp/data/celeba64/{train,val,test}.npy`（合计约 2.5 GB）和
`preview.png`、`meta.json`。

### 步骤 6：冒烟测试（几分钟）

```bash
bash smoke_test.sh
```

看到 **「冒烟测试通过」** 再往下走。它会自动找到装了 torch 的 python，
不用手动指定 `PY`。跑完清理：

```bash
rm -rf _smoke
```

### 步骤 7：正式训练（放后台）

```bash
tmux new -s celeba
```

在 tmux 里：

```bash
cd ~/server

python train_gan.py --data /root/autodl-tmp/data/celeba64 \
    --out /root/autodl-tmp/runs/gan64 \
    --epochs 30 --batch-size 128 --amp --save-every 5 \
    2>&1 | tee /root/autodl-tmp/gan.log

python train_vae.py --data /root/autodl-tmp/data/celeba64 \
    --out /root/autodl-tmp/runs/vae64 \
    --epochs 30 --batch-size 128 --lr 1e-3 --kl-warmup 5 --amp --save-every 5 \
    2>&1 | tee /root/autodl-tmp/vae.log
```

按 `Ctrl+B` 然后按 `D` 退出 tmux，训练继续在后台跑。
回来查看：`tmux attach -t celeba`。

> 想一条命令跑完全程（数据已转好时）：
> ```bash
> ROOT=/root/autodl-pub/CelebA FULL=1 \
>   DATA=/root/autodl-tmp/data/celeba64 \
>   GAN_OUT=/root/autodl-tmp/runs/gan64 \
>   VAE_OUT=/root/autodl-tmp/runs/vae64 \
>   EVAL_OUT=/root/autodl-tmp/runs/eval64 \
>   bash run_all.sh 2>&1 | tee /root/autodl-tmp/all.log
> ```

### 步骤 8：评估

```bash
cd ~/server

python evaluate.py \
    --data /root/autodl-tmp/data/celeba64 \
    --out  /root/autodl-tmp/runs/eval64 \
    --gan  /root/autodl-tmp/runs/gan64 \
    --vae  /root/autodl-tmp/runs/vae64 \
    --split test --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
```

`--fid` 首次会下载 Inception-v3（约 100 MB），不想等就去掉这个参数。

### 步骤 9：取回结果

```bash
# 本地执行
scp -r root@你的实例IP:/root/autodl-tmp/runs/eval64 ./runs_eval
```

报告要用的就是 `runs_eval/` 里的 `report.md`、`results.json` 和几张
`compare_*.png`，详见第 7 节。

### 全部命令连成一段（数据已转好的情况下）

```bash
ssh root@你的实例IP
cd ~ && tar -xzf server_bundle.tar.gz && cd server
bash smoke_test.sh && rm -rf _smoke
tmux new -s celeba
# —— 进入 tmux 后 ——
ROOT=/root/autodl-pub/CelebA FULL=1 \
  DATA=/root/autodl-tmp/data/celeba64 \
  GAN_OUT=/root/autodl-tmp/runs/gan64 \
  VAE_OUT=/root/autodl-tmp/runs/vae64 \
  EVAL_OUT=/root/autodl-tmp/runs/eval64 \
  bash run_all.sh 2>&1 | tee /root/autodl-tmp/all.log
# Ctrl+B 然后 D 退出
```

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

## 3. 准备数据（四选一）

### 方案 0：直接用 AutoDL 公共数据集（最省事，推荐）

AutoDL 的公共数据集已经挂载在 `/root/autodl-pub`（只读），CelebA 就在里面，
官方目录结构：`Anno/ Eval/ Img/ README.txt`，其中
`Img/img_align_celeba.zip` 是图片、`Eval/list_eval_partition.txt` 是官方划分。
**不用上传、不用解压**，脚本直接从 zip 里读 jpg：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --size 64
```

先 dry-run 确认检测到的路径对不对（不写文件）：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --dry-run
# [0/4] 扫描官方目录 /root/autodl-pub/CelebA ...
#       图片源: 压缩包 Img/img_align_celeba.zip（无需解压，直接读）
#       划分表: /root/autodl-pub/CelebA/Eval/list_eval_partition.txt
#       共 202599 张图片
```

输出必须写到 **`/root/autodl-tmp`** 或数据盘，不要写到系统盘（30 G 会满）：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba64 --size 64
```

### 方案 A：传 zip，在服务器上转换

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

# 划分表在别处时一起指定
python prepare_data.py --src /path/to/img_align_celeba \
    --partition /path/to/Eval/list_eval_partition.txt --out data/celeba64
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
PY=$(which python) bash smoke_test.sh  # 强制用当前环境的 python
```

脚本默认会自动探测解释器（优先带 CUDA 的），一般不用指定 `PY`。

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

# AutoDL 上用公共数据集，输出写到数据盘
ROOT=/root/autodl-pub/CelebA FULL=1 \
  DATA=/root/autodl-tmp/data/celeba64 \
  GAN_OUT=/root/autodl-tmp/runs/gan64 \
  VAE_OUT=/root/autodl-tmp/runs/vae64 \
  EVAL_OUT=/root/autodl-tmp/runs/eval64 \
  bash run_all.sh 2>&1 | tee train.log
```

> `run_all.sh` 检测到 `/root/autodl-pub/CelebA` 存在时会自动用它，
> 但**输出目录仍默认在项目内**，注意别把 2.5 GB 的 npy 和几个 G 的
> checkpoint 写进只有 30 G 的系统盘。

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

**AutoDL：`/root/autodl-pub` 是只读的**
公共数据集只读挂载，所以 `--out` 一定要指向 `/root/autodl-tmp/...`，
否则会报 `Read-only file system` 或 `Permission denied`。

**AutoDL：`Img/img_align_celeba.zip` 到底解不解压**
不用解压。`--root /root/autodl-pub/CelebA` 会直接从 zip 里读，
比先解压再转换省 1.5 GB 磁盘和一次全量 IO。只有 `img_align_celeba_png.7z`
和 `img_celeba.7z` 是 7z 格式，脚本不支持，但也不用它们（zip 里就是同样的 jpg）。

**生成结果全是噪声**
先确认训练轮数够不够（GAN 至少 20~30 epoch 才有形状），
再看 `runs/gan64/loss.csv` 里 D 和 G 的损失是否还在正常范围。

**GPU 占用只有 50%、显存只用了 1 GB，是不是没跑满？**

正常。64px 的 DCGAN 只有 5.65 M 参数（G 2.88 M + D 2.77 M），
权重 + Adam 状态 + EMA 不到 100 MB，加 CUDA context 和激活值，
1 GB 出头就是全部家当。4090D 的 24 GB 有 23 GB 闲着，是模型形状决定的。

占用 50% 是因为这个分辨率下 kernel 只有几十微秒，GPU 大部分时间在等
下一个 kernel 派发（launch-bound），不是数据没喂上。判断方法：

```bash
watch -n 1 nvidia-smi
```

- 占用 30~60% 稳定不抖 → launch-bound，正常
- 占用 5~20% 剧烈抖动、`top` 里 python CPU 很高 → data-bound，加 `--num-workers`
- 占用 90%+ → 已经满了

按 40 秒/轮算，162079 张 ÷ 1266 step ≈ 31.6 ms/step ≈ 4000 张/秒，
30 轮 20 分钟跑完。**这个速度对 64px DCGAN 是正常的。**

真想压榨显卡，按性价比：

```bash
# 1) 加大 batch（最省事），注意 lr 要跟着调
--batch-size 512 --lr 4e-4

# 2) 升到 128 分辨率（真正吃满算力，但慢 4~6 倍）
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba128 --size 128 --workers 8
--batch-size 128 --ngf 128 --ndf 128 --amp
```

完整说明见项目 README 第 2.1 节「性能调优」。
**但对这个实验来说，占用 50% 还是 90% 不影响最终的 PSNR/SSIM 数字。**

**指标和 README 里的预期差很多**
README 第 5 节的区间是 30 epoch 量级的经验值，实际受数据量、超参影响。
以 `results.json` 为准，并在报告里说明训练配置。

---

## 附：这个包里有什么

| 文件 | 作用 |
| --- | --- |
| `prepare_data.py` | 官方目录/zip/文件夹 → train/val/test.npy（官方划分） |
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
