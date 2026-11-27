# PyTorch-CelebA-Faces —— GAN 与 VAE 人脸图像生成

河海大学《机器学习（项目式）》实验十：深度生成网络实验，基于 CelebA 数据集的人脸图像生成。
分别训练 **DCGAN** 与 **ConvVAE**，采用 **PSNR / SSIM**（可选 FID）进行定量评估，
并通过生成图像与真实图像的对照进行定性评估。

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
├── run_all.sh          # 一键执行完整流程
├── smoke_test.sh       # 冒烟测试（环境/数据/模型/评估全链路）
├── collect_results.sh  # 收集训练与评估产物，打包下载
└── pack_server.sh      # 归集服务器运行包到 server/
```

## 0. 环境

```bash
conda create -n celeba python=3.10 -y
conda activate celeba
# 服务器按 CUDA 版本安装（示例 CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

本项目的网络规模较小（64×64 输入），CPU 亦可运行，但训练速度显著低于 GPU。
程序默认自动选择 `cuda` 设备。

## 1. 数据准备

脚本支持三种数据来源。其中 AutoDL 公共数据集可直接使用，无需解压或上传。

### 1.1 AutoDL 公共数据集（推荐，服务器环境）

AutoDL 的 `/root/autodl-pub/CelebA` 为标准官方目录结构：

```
CelebA/
├── Anno/  list_attr_celeba.txt  list_bbox_celeba.txt  ...
├── Eval/  list_eval_partition.txt          ← 官方 train/val/test 划分
├── Img/
│   ├── img_align_celeba.zip                ← 图片（未解压，脚本直接读取）
│   ├── img_align_celeba_png.7z*
│   └── img_celeba.7z*
└── README.txt
```

指定根目录即可，脚本会自动定位 `Img/img_align_celeba.zip` 与 `Eval/list_eval_partition.txt`：

```bash
# 预检：仅输出检测结果，不实际转换
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --dry-run

# 正式转换（zip 免解压直读，20 万张 16 进程约 5~15 分钟）
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64 --size 64

# 快速验证：仅取 5000 张
python prepare_data.py --root /root/autodl-pub/CelebA --out data/celeba64_5k \
    --size 64 --max-images 5000
```

> 无需解压 `img_align_celeba.zip`，脚本直接从压缩包内读取 jpg，
> 可节省 1.5 GB 磁盘空间与解压时间。`img_align_celeba.7z` / `img_celeba.7z`
> 为 7z 格式，脚本不支持；若公共数据仅提供 7z，程序会打印对应的解压命令。

### 1.2 Kaggle 压缩包

将 `archive.zip`（内含 `img_align_celeba/img_align_celeba/*.jpg` 与
`list_eval_partition.csv`）置于项目根目录：

```bash
python prepare_data.py --zip archive.zip --out data/celeba64 --size 64
```

### 1.3 已解压的图片文件夹

```bash
python prepare_data.py --src ./img_align_celeba --out data/celeba64 --size 64
# 若划分表位于其他位置，使用 --partition 指定
python prepare_data.py --src ./img_align_celeba \
    --partition ./Eval/list_eval_partition.txt --out data/celeba64
```

### 输出

`data/celeba64/{train,val,test}.npy` 以及 `meta.json`、`preview.png`。
训练前应查看 `preview.png`，确认人脸居中、无黑边。
若未找到官方划分表，程序按 8:1:1 稳定哈希划分作为兜底，
具体采用何种方式记录在 `meta.json` 的 `split_source` 字段中。

## 2. 训练

```bash
# DCGAN（30 epoch，batch 128；4090D 约 20 分钟，4060 笔记本约 1 小时）
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 128 --lr 2e-4 --amp

# ConvVAE（轮数相同，速度快于 GAN）
python train_vae.py --data data/celeba64 --out runs/vae64 \
    --epochs 30 --batch-size 128 --lr 1e-3 --latent-dim 128 --beta 1.0 --kl-warmup 5 --amp
```

训练产物包括 `last.pt` / `best.pt` / `epoch_XXX.pt`、`loss.csv`、
`samples/`（每轮生成图）与 `hparams.json`。断点续训：`--resume runs/gan64/last.pt`。

### 常用调参

| 目标 | 措施 |
| --- | --- |
| GAN 训练不稳定、D 损失迅速归零 | 启用 `--spectral-norm`；或改用 `--loss hinge`；或降低 D 的学习率 |
| GAN 生成图像模糊 | 增加训练轮数；`--ngf 64 --ndf 64` 为 64×64 的常规配置，不宜盲目加宽通道数 |
| GAN 模式崩塌（样本趋同） | 保持 `--ema-decay 0.999`；减少 `--n-critic`；适当降低学习率 |
| VAE 重建图像模糊 | 降低 `--beta`（如 0.5）或改用 `--recon l1` |
| VAE 后验塌缩（KL≈0） | 使用 `--kl-warmup 5~10`；或减小 `--latent-dim` |

## 2.1 性能调优

### 瓶颈定位

训练启动后，另开终端执行：

```bash
watch -n 1 nvidia-smi
```

重点关注 **GPU-Util**（算力占用率）与 **Memory-Used**（显存占用）两列。
根据观测结果按下表判断瓶颈类型：

| 观测现象 | 瓶颈类型 | 处理方式 |
| --- | --- | --- |
| 占用率 30%~60%，波动平缓 | **launch-bound**：单次 kernel 执行时间短，GPU 主要耗时在等待 kernel 派发 | 增大 batch size 或提高分辨率 |
| 占用率 5%~20%，波动剧烈，`top` 中 python 进程 CPU 占用高 | **data-bound**：数据加载速度不足 | 增大 `--num-workers`；确认 `.npy` 位于本地磁盘 |
| 占用率 90% 以上，显存接近上限 | 算力已充分利用 | 无需调整；必要时降低 batch size 以避免 OOM |
| 显存占用约 1 GB、占用率约 50% | 正常状态 | 该模型的显存需求本即如此 |

**显存占用低并不等同于资源浪费。** 64×64 的 DCGAN 共 5.65 M 参数
（生成器 2.88 M + 判别器 2.77 M），权重、Adam 状态与 EMA 副本合计不足 100 MB；
叠加 CUDA context 与激活值后，1 GB 左右即为该模型的全部显存开销。
24 GB 显卡上剩余约 23 GB 空闲，由模型规模决定，并非参数配置不当。

**另一类常见误判**在于对占用率的解读。`torch.cuda.Util` 等工具报告的是
采样时刻是否存在 kernel 正在执行；由于 kernel 执行时间仅数十微秒且之间存在空隙，
采样结果易在 50% 上下波动。真实利用率应参考 `nvidia-smi dmon` 的统计均值，
或直接以端到端指标（每轮训练耗时）衡量。

### 四个主要优化方向（按性价比排序）

#### ① 增大 batch size

batch size 越大，单次 kernel 的计算量越充分，固定启动开销被摊薄的程度越高。

```bash
# 64px 下 batch 512 约占用 2~3 GB 显存
python train_gan.py --data data/celeba64 --out runs/gan64 \
    --epochs 30 --batch-size 512 --lr 4e-4 --amp --save-every 5
```

**增大 batch size 后须同步调整学习率。** 常用做法为线性缩放：
batch size 由 128 增至 512（4 倍）时，`--lr` 取 8e-4 为激进方案，取 4e-4 为保守方案。
GAN 训练建议先用 4e-4，若 D 损失震荡剧烈则退回 2e-4。
仅增大 batch size 而不调整学习率，会导致收敛速度明显下降。

#### ② 提高分辨率

64×64 输入下特征图尺寸最大为 64×64，卷积核难以占满单个 SM。
提升至 128×128 后，特征图面积扩大 4 倍、参数量约扩大 10 倍，
显存占用可达 4~6 GB，占用率随之上升。

```bash
# 先生成 128 分辨率数据（约 8 GB npy，需注意磁盘容量）
python prepare_data.py --root /root/autodl-pub/CelebA \
    --out /root/autodl-tmp/data/celeba128 --size 128 --workers 8

python train_gan.py --data /root/autodl-tmp/data/celeba128 \
    --out /root/autodl-tmp/runs/gan128 \
    --epochs 30 --batch-size 128 --ngf 128 --ndf 128 --amp --save-every 5
```

**需权衡收益与代价**：单轮耗时增加 4~6 倍，30 轮由约 20 分钟延长至 1.5~2 小时。
收益在于 128×128 下五官细节清晰可辨，而 64×64 输出的面部细节较为模糊。
若实验报告需要呈现对比图，128×128 的观感明显更优；
若仅用于验证 GAN 与 VAE 的方法差异，64×64 已足够。

#### ③ 混合精度 `--amp`

`run_all.sh` 中 GAN 与 VAE 均已启用 `--amp`，将卷积与矩阵乘运算置于 fp16 下执行，
理论加速比 2 倍、显存占用减半。该项已默认开启，无需另行设置，
是本仓库中唯一必须启用的性能选项。

#### ④ DataLoader 的 `--num-workers`

默认值为 4。若 `nvidia-smi` 显示占用率低于 30% 且波动剧烈，应优先检查此项：

```bash
--num-workers 8      # 数据位于本地 SSD 时
--num-workers 0      # 容器禁止多进程时（速度下降，但可保证运行）
```

AutoDL 的 `/root/autodl-tmp` 为本地盘，8 个 worker 一般可正常工作。
`/root/autodl-pub` 为网络挂载，不应直接从中读取图片进行训练，
须先经 `prepare_data.py` 转换为 npy 文件。

### 已默认生效的优化

以下优化项已在代码中默认启用，无需额外配置：

| 优化项 | 位置 | 作用 |
| --- | --- | --- |
| `pin_memory=True` | `dataset.py` | CPU→GPU 拷贝异步化，配合 `non_blocking=True` |
| `persistent_workers=True` | `dataset.py` | worker 进程不逐轮重建，节省 fork 开销 |
| `prefetch_factor=2` | `dataset.py` | 预取 2 个 batch |
| `cudnn.benchmark=True` | `utils.py:48` `setup_cudnn()` | 输入尺寸固定时自动选择最快卷积算法 |
| `zero_grad(set_to_none=True)` | 两个训练脚本 | 减少一次显存写操作，小模型上亦可节省 2%~3% |
| D 步生成器前向包裹 `torch.no_grad()` | `train_gan.py:211` | D 步无需 G 的计算图，节省约 40% 显存 |
| EMA 使用独立 `sampler` 模型 | `train_gan.py:189` | 采样不污染训练模型的 BN 统计量 |

### 进一步优化（可选，收益递减）

**消除每步的 `.item()` 同步。** `train_gan.py:237-238` 每步调用两次
`loss_d.item()` / `loss_g.item()`，每次均强制 CUDA 同步等待 GPU 计算完成，
打断了 kernel 的连续派发流水线。改为在 GPU 端累加、每 50 步同步一次，
可节省 5%~10% 耗时。但该改动涉及训练主循环，不建议在正式训练前修改。

**`torch.compile`。** 在小模型上收益较为明显（算子融合、减少启动次数），
但首次编译需 1~2 分钟，且在 Windows 平台编译成功率较低。

```python
G = torch.compile(G)   # 添加在 train_gan.py 第 127 行之后
D = torch.compile(D)
```

风险：可能触发 `TorchDynamo` 相关错误；checkpoint 的 key 名会带 `_orig_mod.` 前缀，
导致 `evaluate.py` 加载失败。启用前应先运行冒烟测试，确认完整链路可用。

### 性能优化的适用边界

本实验的设定为 64×64 CelebA、30 轮训练，输出一组 PSNR/SSIM 对比图与结论。
按单轮约 40 秒计，全量 30 轮约 20 分钟即可完成。
**GPU 占用率为 50% 或 90%，对最终 PSNR/SSIM 数值不产生任何影响。**
相较于优化 GPU 占用率，将时间投入生成质量评估与报告撰写更为有效。

仅在以下情形下，性能调优才具备必要性：

- 单轮耗时超过 5 分钟（需先排查是否使用了 `--num-workers 0` 或运行于 CPU）
- 显存 OOM（此时应反向降低 batch size）
- 需要提高分辨率以获得更高质量的对比图

## 3. 评估

```bash
python evaluate.py --data data/celeba64 --out runs/eval \
    --gan runs/gan64 --vae runs/vae64 \
    --split test --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
```

生成的产物如下：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 全部逐图指标与运行配置 |
| `report.md` | 自动生成的中文结论草稿（数值来自真实计算） |
| `compare_vae_recon.png` | 上排 GT / 下排 VAE 重建 |
| `compare_gan_inversion.png` | 上排 GT / 下排 GAN 反演重建 |
| `compare_gan_nn.png` | 上排最近邻真实图 / 下排 GAN 生成图 |
| `samples_gan.png`、`samples_vae_prior.png`、`samples_vae_recon.png` | 随机采样结果 |
| `metric_bar.png` | PSNR/SSIM 柱状图 |

### 指标口径（报告中须明确说明）

1. **VAE 重建 PSNR/SSIM**：输入真实图像 x，比较重建结果 x̂ 与 x，GT 即 x 本身，
   二者一一对应，口径最为标准。
2. **GAN 随机生成的最近邻 PSNR/SSIM**：GAN 为无条件生成，输出与任何真实图像均无对应关系。
   此处为每张生成图在测试集中检索像素空间最相近的真实图作为"伪 GT"，
   **仅能说明生成分布与真实数据的接近程度，不能作为重建质量指标**。
3. **GAN 潜在反演 PSNR/SSIM**（可选）：为每张真实图优化潜变量 z 使 G(z)≈x，
   得到与 VAE 同口径的重建指标，反映生成器对真实图像的覆盖程度。
4. **FID**：分布级指标，数值越低越好；需要不少于 1000 张样本，
   并依赖 ImageNet 预训练的 Inception-v3。

## 4. 运行方式

### 4.0 部署到服务器

将服务器所需的文件归集到一个自包含目录：

```bash
bash pack_server.sh          # 生成 server/（含 README、VERSION、全部代码）
bash pack_server.sh --tar    # 额外打包为 server_bundle.tar.gz（约 42 KB）
```

随后执行 `scp -r server user@server:~/celeba` 或 `scp server_bundle.tar.gz user@server:~/`。
服务器端的完整流程（环境配置、数据传输、tmux 训练、结果取回、常见问题）
见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。

> `server/` 为生成物，已列入 `.gitignore`。修改代码请编辑项目根目录的源文件，
> 再重新执行打包脚本。

### 4.1 冒烟测试（强烈建议）

在新机器（尤其是服务器）上正式训练前，应先运行冒烟测试，
确认环境、数据、模型、评估四条链路均可用：

```bash
bash smoke_test.sh
```

该脚本仅训练数十步，数分钟内结束，依次检查：

1. torch / torchvision / CUDA 是否可用，显卡型号与显存
2. 数据能否正确读取，形状与 dtype 是否符合预期，是否存在 `preview.png`
3. DataLoader 多进程是否受环境限制（若受限则自动降级为 `num_workers=0`）
4. DCGAN 能否完成前向与反向传播、能否保存检查点
5. ConvVAE 能否完成前向与反向传播、能否保存检查点
6. 评估脚本能否完整执行并产出全部图像

全部通过后输出「冒烟测试通过」。产物位于 `_smoke/`，查看后可执行 `rm -rf _smoke` 删除。

可配置项（均为环境变量）：

```bash
N=5000 STEPS=200 bash smoke_test.sh    # 增加数据量与步数
SIZE=128 bash smoke_test.sh            # 同时验证 128 分辨率
WORKERS=4 bash smoke_test.sh           # 强制多进程（默认自动探测）
DATA_SRC=./img_align_celeba bash smoke_test.sh   # 使用已解压的图片目录
PY=/path/to/python bash smoke_test.sh  # 指定解释器
```

> 冒烟测试仅训练数十步，`results.json` 中的 PSNR/SSIM **不具备参考价值**，
> 其作用仅在于验证流程完整性。

### 4.2 正式训练

```bash
bash run_all.sh            # 数据准备(5k 子集) -> 训练 -> 评估
FULL=1 bash run_all.sh     # 全量数据 + 30 epoch
EPOCHS=50 BATCH=128 bash run_all.sh
```

### 4.3 收集结果

训练与评估完成后，将产物整理成一个便于下载的压缩包：

```bash
bash collect_results.sh
```

脚本会自动探测 `runs/` 下的 GAN、VAE 与评估目录，输出
`results_<时间戳>.tar.gz`（通常 1~10 MB）。包内除图表外还包含：

| 文件 | 内容 |
| --- | --- |
| `README.txt` | 结果说明与指标口径，可直接用于撰写报告 |
| `MANIFEST.txt` | 文件清单（大小 + md5），用于校验传输完整性 |
| `env.txt` | 运行环境（GPU 型号、torch 与 CUDA 版本） |
| `VERSION` | 打包时的代码版本 |

常用选项：

```bash
bash collect_results.sh --with-ckpt             # 附带 last.pt / best.pt
bash collect_results.sh --data data/celeba64    # 附带 meta.json 与 preview.png
bash collect_results.sh --gan runs/gan64 --vae runs/vae64 --eval runs/eval
```

检查点体积较大（每个 70~90 MB），仅在需要复现或继续训练时取回；
评估图表与 `report.md` 均不依赖检查点。

## 5. 预期结果（64×64，30 epoch 量级，供对照）

| 方法 | PSNR (dB) | SSIM | 特点 |
| --- | --- | --- | --- |
| VAE 重建 | 20 ~ 24 | 0.65 ~ 0.80 | 指标较高，但图像偏模糊 |
| GAN 反演重建 | 12 ~ 18 | 0.35 ~ 0.55 | 指标较低，但纹理锐利 |
| GAN 随机生成（最近邻） | 10 ~ 15 | 0.20 ~ 0.40 | 无真实对应，仅供参考 |

具体数值取决于训练轮数与超参数配置，以 `results.json` 为准。

## 6. 结论要点（可直接引用至实验报告）

* **VAE 在像素指标上占优**：其训练目标即为逐像素 MSE，与 PSNR/SSIM 的评价口径完全一致。
* **GAN 在感知质量上占优**：对抗损失使输出落在自然图像流形上，细节锐利；
  但其优化目标不包含像素对齐，故 PSNR/SSIM 天然处于劣势——
  这正是"客观指标与人类感知不一致"的典型案例。
* **VAE 具备编码器**，可用于重建、插值与属性编辑；
  **GAN 的潜变量 z 不具备语义保证**，反演需逐图优化。
* 两者均受 64×64 分辨率限制：提升至 128/256 需要更深的网络、更长的训练时间
  以及更稳定的训练技巧（谱归一化、TTUR、DiffAugment 等）。
