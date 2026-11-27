# 服务器端运行包

本目录为**自包含**运行包。将整个 `server/` 上传至服务器并安装依赖后即可直接运行，
无需 git，也不依赖项目中的其他文件。

> **AutoDL 环境**：若公共数据集已包含 CelebA，则 `archive.zip` 与 `data/` 均无需上传。
> `prepare_data.py --root /root/autodl-pub/CelebA` 会直接读取
> `/root/autodl-pub/CelebA/Img/img_align_celeba.zip`，详见第 3 节方案 0。

> **注意**：本目录由项目根目录的 `pack_server.sh` 生成，请勿直接修改其中代码。
> 代码修改应在项目根目录的源文件上进行，随后重新执行 `bash pack_server.sh`。

---

## 0. 完整命令（从零到出结果）

本节按顺序给出完整可执行命令，可直接照此操作，无需先阅读后续章节。
以下命令均假设运行于 **AutoDL** 环境：conda 位于 `/root/miniconda3`，
公共数据集位于 `/root/autodl-pub/CelebA`，输出写入 `/root/autodl-tmp`
（系统盘容量仅 30 GB）。

### 步骤 1：本地打包并上传

```bash
# 在本地项目根目录 PyTorch-CelebA-Faces/ 下执行
bash pack_server.sh --tar
scp server_bundle.tar.gz root@你的实例IP:~/
```

### 步骤 2：服务器端解压

```bash
ssh root@你的实例IP

cd ~
tar -xzf server_bundle.tar.gz
cd server
cat VERSION            # 确认 commit 与本地打包版本一致
```

### 步骤 3：确认 Python 环境

AutoDL 镜像通常已预装 torch，先执行检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 若输出 `True` 及显卡型号，则跳过步骤 4，直接执行步骤 5。
- 若报 `No module named 'torch'`，则执行步骤 4。

### 步骤 4：（仅在缺少 torch 时）配置环境

```bash
conda create -n celeba python=3.10 -y
conda activate celeba

# 依据 nvidia-smi 右上角的 CUDA Version 选择，此处以 12.1 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 步骤 5：数据转换（约 5~15 分钟）

首先执行 dry-run 确认路径检测结果：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out /root/autodl-tmp/data/celeba64 --dry-run
```

输出中应包含 `图片源: 压缩包 Img/img_align_celeba.zip（无需解压，直接读）`，
确认无误后执行正式转换：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out /root/autodl-tmp/data/celeba64 --size 64 --workers 8
```

产物为 `/root/autodl-tmp/data/celeba64/{train,val,test}.npy`（合计约 2.5 GB），
以及 `preview.png`、`meta.json`。

### 步骤 6：冒烟测试（数分钟）

```bash
bash smoke_test.sh
```

待输出 **「冒烟测试通过」** 后再继续后续步骤。脚本会自动定位已安装 torch 的
Python 解释器，无需手动指定 `PY`。测试完成后清理产物：

```bash
rm -rf _smoke
```

### 步骤 7：正式训练（后台运行）

```bash
tmux new -s celeba
```

在 tmux 会话中执行：

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

按 `Ctrl+B` 后按 `D` 退出 tmux 会话，训练进程将在后台继续运行。
重新接入会话：`tmux attach -t celeba`。

> 若数据已转换完毕，可用单条命令执行完整流程：
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

`--fid` 首次执行时会下载 Inception-v3 权重（约 100 MB）；若无需该项指标可省略此参数。

### 步骤 9：取回结果

```bash
# 服务器端：打包。脚本会自动向外查找 ~/autodl-tmp/runs 下的 GAN、VAE 与评估目录
bash ~/server/collect_results.sh

# 本地执行：下载并解压
scp root@你的实例IP:~/server/results_*.tar.gz ./
tar -xzf results_*.tar.gz
```

压缩包内含 `report.md`、`results.json`、全部 `compare_*.png` 与一份
说明指标口径的 `README.txt`，详见第 7 节。

### 完整命令序列（数据已转换的情况下）

```bash
ssh root@你的实例IP
cd ~ && tar -xzf server_bundle.tar.gz && cd server
bash smoke_test.sh && rm -rf _smoke
tmux new -s celeba
# —— 进入 tmux 会话后执行 ——
ROOT=/root/autodl-pub/CelebA FULL=1 \
  DATA=/root/autodl-tmp/data/celeba64 \
  GAN_OUT=/root/autodl-tmp/runs/gan64 \
  VAE_OUT=/root/autodl-tmp/runs/vae64 \
  EVAL_OUT=/root/autodl-tmp/runs/eval64 \
  bash run_all.sh 2>&1 | tee /root/autodl-tmp/all.log
# 按 Ctrl+B 后按 D 退出

# 训练与评估完成后取回结果
bash ~/server/collect_results.sh
```

---

## 1. 上传到服务器

在**本地**项目根目录执行：

```bash
# 方式一：直接传输目录（改动较小时适用）
scp -r server user@server:~/celeba

# 方式二：先打包再传输（文件较多或网络较慢时更高效）
bash pack_server.sh                      # 生成 server_bundle.tar.gz
scp server_bundle.tar.gz user@server:~/
ssh user@server 'tar -xzf server_bundle.tar.gz'
```

服务器端目录结构如下：

```
~/celeba/
├── README.md            ← 即本文档
├── VERSION              ← 打包时的 commit 与时间
├── smoke_test.sh        ← 冒烟测试
├── run_all.sh           ← 一键完整流程
├── requirements.txt
├── prepare_data.py  dataset.py  utils.py  metrics.py  fid.py
├── train_gan.py  train_vae.py  evaluate.py
└── models/
    ├── __init__.py  gan.py  vae.py
```

## 2. 环境配置

```bash
conda create -n celeba python=3.10 -y
conda activate celeba

# 依据服务器实际 CUDA 版本选择，此处以 CUDA 12.1 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

环境验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 须返回 `True`，否则正式训练的耗时将不可接受。

## 3. 数据准备（四种方案任选其一）

### 方案 0：直接使用 AutoDL 公共数据集（最便捷，推荐）

AutoDL 公共数据集挂载于 `/root/autodl-pub`（只读），其中包含 CelebA，
目录结构为官方格式：`Anno/ Eval/ Img/ README.txt`。其中
`Img/img_align_celeba.zip` 为图片数据，`Eval/list_eval_partition.txt` 为官方划分表。
**无需上传、无需解压**，脚本可直接从 zip 内读取 jpg：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --size 64
```

建议先执行 dry-run 确认检测到的路径是否正确（不写入任何文件）：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --dry-run
# [0/4] 扫描官方目录 /root/autodl-pub/CelebA ...
#       图片源: 压缩包 Img/img_align_celeba.zip（无需解压，直接读）
#       划分表: /root/autodl-pub/CelebA/Eval/list_eval_partition.txt
#       共 202599 张图片
```

输出路径须指向 **`/root/autodl-tmp`** 或其他数据盘，不可写入系统盘（30 GB 易被占满）：

```bash
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba64 --size 64
```

### 方案 A：上传 zip，在服务器端转换

```bash
# 本地执行
scp archive.zip user@server:~/celeba/

# 服务器端执行
cd ~/celeba
python prepare_data.py --zip archive.zip --out data/celeba64 --size 64
```

全量 20 万张图片，16 进程约需 5~15 分钟，产出约 2.5 GB 的 npy 文件。

### 方案 B：上传本地已转换的 npy 文件（节省 CPU，但传输量较大）

```bash
# 本地执行（data/celeba64 下三个 npy 合计约 2.5 GB）
scp -r data/celeba64 user@server:~/celeba/data/
```

适用于服务器 CPU 性能较弱，或本地已完成转换、不希望重复转换的情况。
zip 原始数据为 1.4 GB，转换后为 2.5 GB，可依网络条件选择。

### 方案 C：服务器上已有解压后的图片目录

```bash
python prepare_data.py --src /path/to/img_align_celeba --out data/celeba64 --size 64

# 划分表位于其他位置时一并指定
python prepare_data.py --src /path/to/img_align_celeba \
    --partition /path/to/Eval/list_eval_partition.txt --out data/celeba64
```

### 转换完成后检查预览图

```bash
ls -lh data/celeba64/
# 将 preview.png 取回本地查看，确认人脸居中、无黑边
```

```bash
# 本地执行
scp user@server:~/celeba/data/celeba64/preview.png .
```

## 4. 冒烟测试

```bash
bash smoke_test.sh
```

脚本在数分钟内依次检查：CUDA 是否可用、数据能否正确读取、
DataLoader 多进程是否受环境限制、GAN 与 VAE 能否完成前向反向并保存检查点、
评估能否产出全部图像。

待输出「冒烟测试通过」后再继续后续步骤。产物位于 `_smoke/`，检查完毕后执行
`rm -rf _smoke` 删除。

可配置项（均为环境变量）：

```bash
N=5000 STEPS=200 bash smoke_test.sh    # 增加数据量与训练步数
SIZE=128 bash smoke_test.sh            # 同时验证 128 分辨率
WORKERS=4 bash smoke_test.sh           # 强制多进程
PY=$(which python) bash smoke_test.sh  # 强制使用当前环境的 Python
```

脚本默认自动探测解释器（优先选择 CUDA 可用的），通常无需指定 `PY`。

## 5. 正式训练

训练耗时从数十分钟到数小时不等，**须置于 tmux / screen / nohup 中运行**，
否则 SSH 连接断开时进程会被终止。

```bash
tmux new -s celeba
conda activate celeba
cd ~/celeba
FULL=1 bash run_all.sh 2>&1 | tee train.log
# 按 Ctrl+B 后按 D 退出，训练在后台继续运行
# 重新接入：tmux attach -t celeba
```

若不使用 tmux：

```bash
nohup bash run_all.sh > train.log 2>&1 &
tail -f train.log        # Ctrl+C 仅停止查看日志，不影响训练进程
```

`run_all.sh` 依次执行：数据准备 → DCGAN 训练 → ConvVAE 训练 → 评估。

常用参数组合：

```bash
bash run_all.sh                          # 5k 子集 + 20 epoch，用于快速验证
FULL=1 bash run_all.sh                   # 全量数据 + 30 epoch，用于正式结果
EPOCHS=50 BATCH=128 FULL=1 bash run_all.sh

# AutoDL 环境下使用公共数据集，输出写入数据盘
ROOT=/root/autodl-pub/CelebA FULL=1 \
  DATA=/root/autodl-tmp/data/celeba64 \
  GAN_OUT=/root/autodl-tmp/runs/gan64 \
  VAE_OUT=/root/autodl-tmp/runs/vae64 \
  EVAL_OUT=/root/autodl-tmp/runs/eval64 \
  bash run_all.sh 2>&1 | tee train.log
```

> `run_all.sh` 检测到 `/root/autodl-pub/CelebA` 存在时会自动使用该数据源，
> 但**输出目录仍默认为项目内路径**。须注意避免将 2.5 GB 的 npy 文件与
> 数 GB 的检查点写入容量仅 30 GB 的系统盘。

### 分步执行与中途调参

```bash
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --amp --save-every 5

python train_vae.py --data data/celeba64 --out runs/vae64 \
    --epochs 30 --batch-size 128 --lr 1e-3 --kl-warmup 5 --amp --save-every 5
```

`--save-every 5` 用于控制检查点数量：单个检查点约 70~90 MB
（含优化器状态与 EMA），若每轮均保存，30 轮将占用 2~3 GB，易导致磁盘空间不足。

### 中断后的恢复

```bash
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --amp --resume runs/gan64/last.pt
```

`--resume` 将恢复模型权重、优化器状态、学习率调度器、EMA 及当前 epoch。

### 升级代码后的重跑范围

代码更新后需要重跑哪些步骤，取决于改动是否影响已保存的权重：

| 改动 | 需重跑 |
| --- | --- |
| 仅 `evaluate.py` / `metrics.py`（指标口径、存图、报告） | 仅评估 |
| `models/vae.py` 的损失函数或 `train_vae.py` | VAE 从头训练 + 评估 |
| `models/gan.py` 或 `train_gan.py` | GAN 从头训练 + 评估 |
| `prepare_data.py` 的预处理参数（如 `--size`） | 数据转换 + 两个模型 + 评估 |

VAE 的损失函数缺陷属于**训练期缺陷**：已塌缩的权重无法通过 `--resume`
或微调恢复，必须删除旧输出目录后从头训练。

## 6. 评估

`run_all.sh` 的最后一步会自动执行评估。单独执行方式如下：

```bash
python evaluate.py --data data/celeba64 --out runs/eval \
    --gan runs/gan64 --vae runs/vae64 --split test \
    --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
```

`--fid` 会额外下载 ImageNet 预训练的 Inception-v3 模型（首次约 100 MB），
若不需要该指标可省略此参数。

评估过程会做两项自检，结果直接打印在终端并写入 `results.json` 的 `diversity` 字段：

- **VAE 重建多样性**：16 张重建两两平均绝对差 ÷ 真实图同值。
  低于 10% 时判定为后验塌缩，终端打印警告，`report.md` 的「多样性」列也会标出。
- **GAN 生成多样性**：同样口径，用于识别模式崩塌。

两者接近 0% 时 PSNR/SSIM 不具备参考价值，应先修复训练问题再讨论指标高低。

## 7. 取回结果

### 7.1 一键打包（推荐）

`collect_results.sh` 会自动定位训练与评估产物，把撰写报告所需的文件整理成
一个带说明的压缩包。在代码目录执行即可，无需先切换到输出目录：

```bash
bash ~/server/collect_results.sh
```

脚本会依次检查脚本所在目录下的 `runs/`、上一级与上两级的 `runs/`，
以及 `~/autodl-tmp/runs`、`/root/autodl-tmp/runs` 等常见位置。
本实验的默认配置下，输出位于 `/root/autodl-tmp/runs/{gan64,vae64,eval64}`，
均可自动识别。

脚本输出形如 `results_20261127_153012.tar.gz`，体积通常在 1~10 MB（取决于样本图数量）。
包内结构：

```
results_<时间戳>/
├── README.txt      结果说明与指标口径（可直接用于撰写报告）
├── MANIFEST.txt    文件清单（大小 + md5，用于校验传输完整性）
├── env.txt         运行环境（GPU 型号、torch 与 CUDA 版本）
├── VERSION         打包时的代码版本
├── eval/           report.md、results.json 与全部对比图
├── gan/            loss.csv、train_summary.json、samples/
├── vae/            同上
└── data/           meta.json 与 preview.png（能自动找到数据目录时附带）
```

常用选项：

```bash
bash collect_results.sh --runs /root/autodl-tmp/runs   # 指定结果根目录
bash collect_results.sh --with-ckpt      # 附带 last.pt / best.pt（每个 70~90 MB）
bash collect_results.sh --with-epochs    # 附带全部 epoch_*.pt（体积很大）
bash collect_results.sh --data data/celeba64   # 附带 meta.json 与 preview.png
bash collect_results.sh --gan runs/gan64 --vae runs/vae64 --eval runs/eval   # 逐项指定
bash collect_results.sh --out 第一轮64px  # 自定义包名
```

若自动探测未命中，脚本会列出已查找过的路径并提示改用 `--runs` 或逐项指定。

随后下载到本地：

```bash
# 本地执行
scp user@server:~/server/results_20261127_153012.tar.gz ./
tar -xzf results_20261127_153012.tar.gz
```

检查点体积较大，仅在需要复现或继续训练时取回。评估所需的图表与
`report.md` 均不依赖检查点。

### 7.2 手动取回

```bash
# 本地执行
scp -r user@server:~/celeba/runs/eval ./runs_eval
```

实验报告所需的文件如下：

| 文件 | 用途 |
| --- | --- |
| `report.md` | 自动生成的中文结论草稿，数值来自真实计算 |
| `results.json` | 全部逐图指标，撰写报告时用于查询具体数值 |
| `compare_vae_recon.png` | 上排 GT / 下排 VAE 重建，用于定性评估 |
| `compare_gan_inversion.png` | 上排 GT / 下排 GAN 反演重建，用于定性评估 |
| `compare_gan_nn.png` | GAN 随机生成图像与最近邻真实图像对照 |
| `samples_gan.png`、`samples_vae_prior.png` | 随机采样结果 |
| `metric_bar.png` | PSNR/SSIM 柱状图 |

训练过程图像（`runs/gan64/samples/`、`runs/vae64/` 下的 `recon_*.png`）
可用于说明训练收敛情况，如需使用可一并取回。

## 8. 常见问题

**CUDA out of memory**
降低 batch size：`--batch-size 64` 或 `32`。64×64 分辨率下 batch 128 约占用
2~3 GB 显存，8 GB 显卡仍有余量；若发生 OOM，应先确认是否有其他进程占用显存。

**`DataLoader worker ... PermissionError` / 启动时卡住**
部分容器环境禁止多进程。`smoke_test.sh` 会自动探测并降级；
手动运行时添加 `--num-workers 0`。

**训练速度慢、GPU 利用率低**
`--num-workers 0` 时数据加载会成为瓶颈。先执行 `nvidia-smi` 查看 GPU 利用率，
若低于 50% 则调大 `--num-workers 4~8`；同时确认数据位于本地磁盘而非网络挂载盘。

**磁盘空间不足**
使用 `--save-every 5` 减少检查点数量；`data/celeba64/*.npy` 使用完毕后可删除并重新生成；
`runs/*/epoch_*.pt` 仅保留最后若干个。

**AutoDL：`/root/autodl-pub` 为只读挂载**
公共数据集以只读方式挂载，因此 `--out` 必须指向 `/root/autodl-tmp/...`，
否则会报 `Read-only file system` 或 `Permission denied`。

**AutoDL：是否需要解压 `Img/img_align_celeba.zip`**
无需解压。`--root /root/autodl-pub/CelebA` 会直接从 zip 内读取，
相比先解压再转换可节省 1.5 GB 磁盘空间与一次全量 IO。
`img_align_celeba_png.7z` 与 `img_celeba.7z` 为 7z 格式，脚本不支持，
但亦无需使用（zip 内含相同的 jpg 数据）。

**VAE 重建 PSNR 只有 11 dB 左右、生成图全是同一张脸**
判定为**后验塌缩**：解码器输出与输入无关，所有重建都退化成同一张"平均脸"。
检查 `report.md` 定量表的「多样性」列，若接近 0%（或日志中出现
`[警告] ... 疑似后验塌缩`），说明重建损失与 KL 的量级失衡。
本仓库 2026-11-27 之前的版本存在此缺陷（`recon_loss_per_sample` 对像素求平均、
而 `kl_per_sample` 对 latent 维求和，KL 比重建项大三个数量级），
已修复；若服务器上运行的是旧版代码，重新打包上传并**从头训练** VAE
（塌缩后的权重无法通过微调恢复）。

**GAN 反演重建 PSNR 只有 3 dB、对比图下排发白**
旧版 `evaluate.py` 把 `[0,1]` 的预测与 `[-1,1]` 的 GT 直接比较，
且 `save_pair_grid` 会把整张拼图再归一化一次，导致数值偏低约 16 dB、
图像被压到 `[0.5,1]`。已修复；仅需重跑评估，无需重新训练 GAN。

**生成结果全为噪声**
先确认训练轮数是否充足（GAN 通常需 20~30 epoch 才能生成可辨识形状），
再检查 `runs/gan64/loss.csv` 中 D 与 G 的损失是否处于正常范围。

**GPU 占用率约 50%、显存占用约 1 GB，是否未充分利用硬件**

属正常现象。64×64 的 DCGAN 仅 5.65 M 参数（生成器 2.88 M + 判别器 2.77 M），
权重、Adam 状态与 EMA 合计不足 100 MB；叠加 CUDA context 与激活值后，
1 GB 左右即为全部显存开销。4090D 的 24 GB 显存中约 23 GB 空闲，由模型规模决定。

占用率 50% 的原因在于该分辨率下 kernel 执行时间仅数十微秒，
GPU 主要耗时于等待下一个 kernel 派发（launch-bound），并非数据加载不足。判断方法：

```bash
watch -n 1 nvidia-smi
```

- 占用率 30%~60%，波动平缓 → launch-bound，属正常
- 占用率 5%~20%，波动剧烈且 `top` 中 python CPU 占用高 → data-bound，增大 `--num-workers`
- 占用率 90% 以上 → 算力已充分利用

按单轮约 40 秒计：162079 张 ÷ 1266 step ≈ 31.6 ms/step ≈ 4000 张/秒，
30 轮约 20 分钟完成。**该速度对 64×64 DCGAN 而言属正常水平。**

若确需提升硬件利用率，可按性价比顺序考虑：

```bash
# 1) 增大 batch size（最简便），须同步调整学习率
--batch-size 512 --lr 4e-4

# 2) 提升至 128 分辨率（可充分利用算力，但单轮耗时增加 4~6 倍）
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba128 --size 128 --workers 8
--batch-size 128 --ngf 128 --ndf 128 --amp
```

完整说明见项目 README 第 2.1 节「性能调优」。
**但对本实验而言，占用率 50% 或 90% 不影响最终 PSNR/SSIM 数值。**

**指标与 README 中的预期值差异较大**
README 第 5 节给出的区间为 30 epoch 量级的经验值，实际结果受数据量与超参数影响。
应以 `results.json` 为准，并在报告中说明具体训练配置。

---

## 附录：运行包内容

| 文件 | 作用 |
| --- | --- |
| `prepare_data.py` | 官方目录/zip/文件夹 → train/val/test.npy（官方划分） |
| `dataset.py` | Dataset / DataLoader 封装 |
| `utils.py` | seed、device、EMA、checkpoint、日志 |
| `metrics.py` | PSNR / SSIM（纯 torch 实现）、最近邻配对 |
| `fid.py` | FID（Inception-v3 特征） |
| `models/gan.py` | DCGAN 生成器与判别器 |
| `models/vae.py` | 卷积 VAE |
| `train_gan.py` | GAN 训练 |
| `train_vae.py` | VAE 训练 |
| `evaluate.py` | 定量与定性评估，自动生成结论 |
| `smoke_test.sh` | 冒烟测试 |
| `run_all.sh` | 一键完整流程 |
| `collect_results.sh` | 收集训练与评估产物，打包下载 |
