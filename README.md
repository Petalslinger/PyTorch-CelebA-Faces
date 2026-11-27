# PyTorch-CelebA-Faces —— GAN 与 VAE 人脸图像生成

河海大学《机器学习（项目式）》实验十：深度生成网络实验，基于 CelebA 数据集的人脸图像生成。
分别训练 **DCGAN** 与 **ConvVAE**，并用 **PSNR / SSIM**（+ 可选 FID）做定量评估、用对比图做定性评估。

```
.
├── prepare_data.py     # 官方目录/zip/文件夹 -> 统一尺寸的 .npy（含官方 train/val/test 划分）
├── dataset.py          # Dataset / DataLoader（mmap 读取，[-1,1] 归一化）
├── utils.py            # seed、device、日志、EMA、checkpoint、网格图
├── metrics.py          # PSNR、SSIM（纯 torch）、最近邻配对
├── fid.py              # FID（Inception-v3 特征）
├── models/
│   ├── gan.py          # Generator / Discriminator（DCGAN，可选谱归一化）
│   └── vae.py          # ConvVAE（beta 可调、KL warm-up、三种重建损失）
├── train_gan.py        # DCGAN 训练
├── train_vae.py        # ConvVAE 训练
├── evaluate.py         # 定量 + 定性 + 自动生成结论草稿
└── run_all.sh          # 一键跑完整流程
```

## 0. 环境

```bash
conda create -n celeba python=3.10 -y
conda activate celeba
# 服务器按 CUDA 版本安装（示例 CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU 也能跑（64x64 的小网络），只是慢；默认自动选择 `cuda`。

## 1. 数据准备

脚本支持三种数据来源，**AutoDL 公共数据集可以直接用**（无需解压、无需上传）。

### 1.1 AutoDL 公共数据集（推荐，服务器上用）

AutoDL 的 `/root/autodl-pub/CelebA` 是官方目录结构：

```
CelebA/
├── Anno/  list_attr_celeba.txt  list_bbox_celeba.txt  ...
├── Eval/  list_eval_partition.txt          ← 官方 train/val/test 划分
├── Img/
│   ├── img_align_celeba.zip                ← 图片（未解压，脚本直接读）
│   ├── img_align_celeba_png.7z*
│   └── img_celeba.7z*
└── README.txt
```

直接指定根目录即可，脚本会自动找到 `Img/img_align_celeba.zip` 和 `Eval/list_eval_partition.txt`：

```bash
# 先看一眼检测结果，不实际转换
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --dry-run

# 正式转换（zip 免解压直读，20 万张 16 进程约 5~15 分钟）
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --size 64

# 快速试跑：只取 5000 张
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64_5k \
    --size 64 --max-images 5000
```

> **不需要解压 `img_align_celeba.zip`**，脚本直接从压缩包里读 jpg，
> 省 1.5 GB 磁盘和解压时间。`img_align_celeba.7z` / `img_celeba.7z` 脚本不支持，
> 如果公共数据只有 7z，会打印解压命令。

### 1.2 Kaggle 压缩包

把 `archive.zip`（内含 `img_align_celeba/img_align_celeba/*.jpg` 和
`list_eval_partition.csv`）放到项目根目录：

```bash
python prepare_data.py --zip archive.zip --out data/celeba64 --size 64
```

### 1.3 已解压的图片文件夹

```bash
python prepare_data.py --src ./img_align_celeba --out data/celeba64 --size 64
# 如果划分表在别处，用 --partition 指定
python prepare_data.py --src ./img_align_celeba \
    --partition ./Eval/list_eval_partition.txt --out data/celeba64
```

### 输出

`data/celeba64/{train,val,test}.npy` + `meta.json` + `preview.png`。
**先看 preview.png**，确认人脸是正的、没有黑边再训练。
没有官方划分表时会按 8:1:1 稳定哈希兜底（`meta.json` 里 `split_source` 会写明用了哪种）。

## 2. 训练

```bash
# DCGAN（30 epoch，128 batch；4090D 上约 20 分钟，4060 笔记本约 1 小时）
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --lr 2e-4 --amp

# ConvVAE（同样的轮数，比 GAN 快）
python train_vae.py --data data/celeba64 --out runs/vae64 \
    --epochs 30 --batch-size 128 --lr 1e-3 --latent-dim 128 --beta 1.0 --kl-warmup 5 --amp
```

训练产物：`last.pt` / `best.pt` / `epoch_XXX.pt`、`loss.csv`、`samples/`（每轮生成图）、
`hparams.json`。断点续训：`--resume runs/gan64/last.pt`。

### 常用调参

| 目的 | 做法 |
| --- | --- |
| GAN 训练不稳、D 损失很快归零 | 加 `--spectral-norm`；或 `--loss hinge`；或降低 D 的学习率 |
| GAN 样本太糊 | 增加 epoch；`--ngf 64 --ndf 64` 是 64x64 的常规配置，别一味加宽 |
| GAN 模式崩塌（样本雷同） | 保持 `--ema-decay 0.999`；减少 `--n-critic`；适当降 lr |
| VAE 重建太糊 | 降 `--beta`（如 0.5）或用 `--recon l1` |
| VAE 后验塌缩（KL≈0） | 用 `--kl-warmup 5~10`；或减小 `--latent-dim` |

## 2.1 性能调优

### 先判断瓶颈在哪，别瞎调

跑起来之后另开一个终端：

```bash
watch -n 1 nvidia-smi
```

看三列：**GPU-Util**（算力占用）、**Memory-Used**（显存）、**GPU 名字**。

| 现象 | 含义 | 该怎么办 |
| --- | --- | --- |
| 占用 30~60%，稳定不抖 | **launch-bound**：GPU 算得很快，大部分时间在等下一个 kernel 派发 | 加大 batch、提高分辨率（见下） |
| 占用 5~20%，剧烈抖动，`top` 里 python 的 CPU 很高 | **data-bound**：数据加载跟不上 | 加 `--num-workers`；确认 `.npy` 在本地盘 |
| 占用 90%+，显存也快满 | 已经吃满了 | 别调了，调小一点防止 OOM |
| 显存只用了 1 GB 但占用 50% | 正常 | 这个模型就只需要这么多显存 |

**关键认知：显存占用低 ≠ 浪费。** 64×64 的 DCGAN 一共 5.65 M 参数（G 2.88 M + D 2.77 M），
权重 + Adam 状态 + EMA 加起来不到 100 MB，加 CUDA context 和激活值，1 GB 出头就是全部家当。
24 GB 的卡有 23 GB 闲着是模型的形状决定的，不是参数没调好。

**另一种常见误判：** `torch.cuda.Util` 这类工具报的占用是"采样时刻是否有 kernel 在跑"。
kernel 只有几十微秒、中间有空隙，采样就容易读到 50% 上下。真实利用率要看
`nvidia-smi dmon` 的平均值，或者直接看"每轮训练要多久"这个端到端指标。

### 真正能提速的四个旋钮（按性价比排序）

#### ① 加大 batch（最简单）

batch 越大，每个 kernel 的活越多，启动开销摊得越薄。

```bash
# 64px 下 batch 512 大约吃 2~3 GB 显存
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 512 --lr 4e-4 --amp --save-every 5
```

⚠️ **batch 变大后学习率要跟着调**。经验做法是线性缩放：batch 从 128 到 512（4 倍），
`--lr` 从 2e-4 到 8e-4 是激进版，4e-4 是保守版。GAN 建议先用 4e-4 试，
如果 D 损失震荡厉害就退回 2e-4。**不要只改 batch 不改 lr**，否则收敛会明显变慢。

#### ② 提高分辨率（真正吃满算力的办法）

64px 的特征图最大才 64×64，卷积核在上面跑不满一个 SM。升到 128px 后
特征图面积 ×4、参数量 ×10，显存能到 4~6 GB，占用自然上去。

```bash
# 先转 128 的数据（约 8 GB npy，注意磁盘）
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba128 --size 128 --workers 8

python train_gan.py --data /root/autodl-tmp/data/celeba128 \
    --out /root/autodl-tmp/runs/gan128 \
    --epochs 30 --batch-size 128 --ngf 128 --ndf 128 --amp --save-every 5
```

**但要想清楚值不值**：每轮慢 4~6 倍，30 轮从 20 分钟变成 1.5~2 小时。
好处是 128px 的人脸五官才看得清，64px 是糊的。**如果报告里要放对比图，128px 观感好得多**；
如果只是验证"GAN 和 VAE 谁能生成人脸"这个方法对比，64px 就够。

#### ③ 混合精度 `--amp`（已经默认开了）

`run_all.sh` 里 GAN 和 VAE 都带 `--amp`。它把卷积和矩阵乘跑在 fp16，
理论 2 倍加速、显存减半。**已经开着就别再动了**，这是本仓库唯一"必须开"的性能选项。

#### ④ DataLoader 的 `--num-workers`

默认 4。如果 `nvidia-smi` 显示占用低于 30% 且抖动，先怀疑这里：

```bash
--num-workers 8      # 数据盘是本地 SSD 时
--num-workers 0      # 容器禁多进程时（会变慢，但至少能跑）
```

AutoDL 的 `/root/autodl-tmp` 是本地盘，8 个 worker 一般没问题。
`/root/autodl-pub` 是网络挂载，**不要**从那里直接读图片训练（要先 `prepare_data.py` 转成 npy）。

### 已经做了的优化（不用你再改）

代码里这些是默认生效的：

| 优化 | 位置 | 效果 |
| --- | --- | --- |
| `pin_memory=True` | `dataset.py` | CPU→GPU 拷贝走异步，配合 `non_blocking=True` |
| `persistent_workers=True` | `dataset.py` | worker 不每轮重建，省 fork 开销 |
| `prefetch_factor=2` | `dataset.py` | 提前取 2 个 batch |
| `cudnn.benchmark=True` | `utils.py:48` `setup_cudnn()` | 输入尺寸固定时自动选最快的卷积算法 |
| `zero_grad(set_to_none=True)` | 两个训练脚本 | 少一次显存写，小模型上也能省 2~3% |
| `torch.no_grad()` 包 D 步的 G 前向 | `train_gan.py:211` | D 步不需要 G 的计算图，省约 40% 显存 |
| EMA 用独立 `sampler` 模型 | `train_gan.py:189` | 采样不污染训练模型的 BN 状态 |

### 想再进一步（可选，收益递减）

**去掉每步的 `.item()` 同步。** `train_gan.py:237-238` 每步调两次
`loss_d.item()` / `loss_g.item()`，每次都会强制 CUDA 同步等 GPU 算完，
打断了"连续派发 kernel"的流水线。改成在 GPU 上累加、每 50 步同步一次能省 5~10%。
但改动会碰训练主循环，**不建议在正式跑之前动**。

**`torch.compile`。** 小模型上收益明显（合并算子、减少启动次数），
但首次编译要 1~2 分钟，而且 Windows 上经常编不过。

```python
G = torch.compile(G)   # 在 train_gan.py 第 127 行之后加
D = torch.compile(D)
```

风险：可能报 `TorchDynamo` 相关错误、checkpoint 的 key 名会带 `_orig_mod.` 前缀导致
`evaluate.py` 加载失败。**要用就先跑冒烟测试确认整条链路能过。**

### 什么时候不该优化

这个实验是 **64px CelebA、30 轮、出一组 PSNR/SSIM 对比图和结论**。
按 40 秒/轮算，全量 30 轮 20 分钟就跑完了。**GPU 占用 50% 还是 90%，对最终的
PSNR/SSIM 数字没有任何影响。** 把时间花在看生成质量、写报告结论上，比调 GPU 占用有价值得多。

只有当出现下面情况时才值得调：

- 一轮超过 5 分钟（说明哪里不对，先查是不是用了 `--num-workers 0` 或 CPU）
- 显存 OOM（反过来，要**降** batch）
- 想升分辨率做更高质量的对比图

## 3. 评估

```bash
python evaluate.py --data data/celeba64 --out runs/eval \
    --gan runs/gan64 --vae runs/vae64 \
    --split test --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
```

生成：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 全部逐图指标 + 配置 |
| `report.md` | 自动生成的中文结论草稿（数字来自真实计算） |
| `compare_vae_recon.png` | 上排 GT / 下排 VAE 重建 |
| `compare_gan_inversion.png` | 上排 GT / 下排 GAN 反演重建 |
| `compare_gan_nn.png` | 上排最近邻真实图 / 下排 GAN 生成图 |
| `samples_gan.png`、`samples_vae_prior.png`、`samples_vae_recon.png` | 随机采样 |
| `metric_bar.png` | PSNR/SSIM 柱状图 |

### 指标口径（写报告时必须说清楚）

1. **VAE 重建 PSNR/SSIM**：输入真实图 x，比较 x̂ 与 x，GT 就是 x 本身 —— 一一对应，最标准。
2. **GAN 随机生成的"最近邻"PSNR/SSIM**：GAN 是无条件生成，输出与任何真实图都没有对应关系。
   这里为每张生成图在测试集里找像素空间最相近的真实图当"伪 GT"，**只能说明生成分布是否靠近真实数据，不能当作重建质量**。
3. **GAN 潜在反演 PSNR/SSIM**（可选）：为每张真实图优化一个 z 使 G(z)≈x，
   得到与 VAE 同口径的重建指标，反映"生成器能覆盖多少真实图像"。
4. **FID**：分布级指标，越低越好；需要 ≥1000 张样本，且依赖 ImageNet 预训练的 Inception。

## 4. 跑起来

### 4.0 部署到服务器

把要上服务器的文件收进一个自包含目录：

```bash
bash pack_server.sh          # 生成 server/（含 README、VERSION、全部代码）
bash pack_server.sh --tar    # 额外打包成 server_bundle.tar.gz（约 35 KB）
```

然后 `scp -r server user@server:~/celeba` 或 `scp server_bundle.tar.gz user@server:~/`。
服务器端的完整流程（装环境、传数据、tmux 训练、取回结果、常见问题）
见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。

> `server/` 是生成物，已在 `.gitignore` 里；改代码请改项目根的文件再重新打包。

### 4.1 先冒烟测试（强烈建议）

换一台新机器（尤其是服务器）先跑这个，确认环境、数据、模型、评估四条链路都通：

```bash
bash smoke_test.sh
```

它只训练几十步，几分钟结束，会依次检查：

1. torch / torchvision / CUDA 是否可用，显卡型号和显存
2. 数据能否读出来、形状和 dtype 是否正确、有没有 preview.png
3. DataLoader 多进程是否被环境限制（被禁则自动降级为 `num_workers=0`）
4. DCGAN 能否前向+反向、能否存检查点
5. ConvVAE 能否前向+反向、能否存检查点
6. 评估脚本能否跑完并产出所有图

全部通过会打印「冒烟测试通过」，产物在 `_smoke/`，看完直接 `rm -rf _smoke`。

可调参数（都是环境变量）：

```bash
N=5000 STEPS=200 bash smoke_test.sh    # 多看一点
SIZE=128 bash smoke_test.sh            # 顺便验证 128 分辨率
WORKERS=4 bash smoke_test.sh           # 强制多进程（默认自动探测）
DATA_SRC=./img_align_celeba bash smoke_test.sh   # 用已解压的图片目录
PY=/path/to/python bash smoke_test.sh  # 指定解释器
```

> 冒烟测试只跑几十步，`results.json` 里的 PSNR/SSIM **没有参考意义**，只能证明流程通。

### 4.2 正式训练

```bash
bash run_all.sh            # 数据准备(5k 子集) -> 训练 -> 评估
FULL=1 bash run_all.sh     # 全量数据 + 30 epoch
EPOCHS=50 BATCH=128 bash run_all.sh
```

## 5. 预期结果（64x64，30 epoch 量级，供对照）

| 方法 | PSNR (dB) | SSIM | 特点 |
| --- | --- | --- | --- |
| VAE 重建 | 20 ~ 24 | 0.65 ~ 0.80 | 指标高，但图像偏模糊 |
| GAN 反演重建 | 12 ~ 18 | 0.35 ~ 0.55 | 指标低，但纹理锐利 |
| GAN 随机生成（最近邻） | 10 ~ 15 | 0.20 ~ 0.40 | 无真实对应，仅作参考 |

具体数值取决于训练轮数与超参，以 `results.json` 为准。

## 6. 结论要点（实验报告可直接引用）

* **VAE 赢在像素指标**：它的训练目标就是逐像素 MSE，与 PSNR/SSIM 完全一致，属于"考什么学什么"。
* **GAN 赢在感知质量**：对抗损失让输出落在自然图像流形上，细节锐利；但它不优化像素对齐，
  所以 PSNR/SSIM 天然吃亏——这正是"指标与人类感知不一致"的经典例子。
* **VAE 有编码器**，可以做重建、插值、属性编辑；**GAN 的 z 没有语义保证**，反演需要逐图优化。
* 二者都受 64x64 分辨率限制：再往上（128/256）需要更深的网络、更长训练和更稳的技巧（谱归一化、TTUR、DiffAugment 等）。
