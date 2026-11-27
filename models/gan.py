"""DCGAN（Radford et al., 2016）在 CelebA 上的实现。

结构遵循原论文的几条经验：
  * 生成器用 ConvTranspose2d + BatchNorm + ReLU，最后一层 Tanh；
  * 判别器用 Conv2d + BatchNorm + LeakyReLU(0.2)，不用池化，用步长卷积下采样；
  * 不使用全连接层（除了 G 的第一层把 z 映射成 4x4 特征图）；
  * 生成器输出范围 [-1, 1]，与 dataset.py 的归一化一致。

另外提供两个在实践里更稳的选项（默认不改变 DCGAN 的原始设定）：
  * spectral_norm=True：判别器用谱归一化，训练更稳定（SAGAN 的做法）；
  * 判别器可选输出 logits（配合 BCEWithLogitsLoss / hinge loss）。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN 论文的初始化：Conv 用 N(0, 0.02)，BN 用 N(1, 0.02)。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """z (B, z_dim) -> 图像 (B, 3, S, S)，取值 [-1, 1]。

    image_size 必须是 2 的幂且 >= 8（4x4 起步，每次上采样 x2）。
    """

    def __init__(self, z_dim: int = 100, ngf: int = 64, image_size: int = 64, nc: int = 3):
        super().__init__()
        if image_size < 8 or (image_size & (image_size - 1)) != 0:
            raise ValueError(f"image_size 必须是 >=8 的 2 的幂，收到 {image_size}")
        self.z_dim = z_dim
        self.image_size = image_size

        n_up = 0
        s = 4
        while s < image_size:
            s *= 2
            n_up += 1  # 64 -> 4 次上采样

        # 4x4 的基础特征图宽度：按上采样次数递减，保证参数量与原论文接近
        base = ngf * (2 ** max(0, 3 - n_up)) if n_up <= 4 else ngf
        base = min(max(base, ngf // 2), ngf * 8)
        widths: List[int] = [base * (2 ** i) for i in range(n_up)]  # 逐层减半
        widths = [min(w, ngf * 8) for w in widths]

        self.fc = nn.ConvTranspose2d(z_dim, widths[0], 4, 1, 0, bias=False)
        self.fc_bn = nn.BatchNorm2d(widths[0])

        blocks = []
        in_ch = widths[0]
        # 中间层：每次分辨率 x2（4->8->...->S/2）
        for i in range(1, len(widths)):
            out_ch = widths[i]
            blocks += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
            in_ch = out_ch
        # 最后一层：S/2 -> S，输出 3 通道并压到 [-1, 1]
        blocks += [
            nn.ConvTranspose2d(in_ch, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*blocks)
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        h = torch.relu(self.fc_bn(self.fc(z)))
        return self.net(h)


class Discriminator(nn.Module):
    """图像 (B,3,S,S) -> 1 个标量（默认是 logits）。"""

    def __init__(
        self,
        ndf: int = 64,
        image_size: int = 64,
        nc: int = 3,
        spectral_norm: bool = False,
        sigmoid: bool = False,
    ):
        super().__init__()
        if image_size < 8 or (image_size & (image_size - 1)) != 0:
            raise ValueError(f"image_size 必须是 >=8 的 2 的幂，收到 {image_size}")

        n_down = 0
        s = image_size
        while s > 4:
            s //= 2
            n_down += 1

        def conv(in_ch, out_ch, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layer = nn.Sequential(*layers)
            return nn.utils.spectral_norm(layer[0]) if spectral_norm else layer

        widths = [min(ndf * (2 ** i), ndf * 8) for i in range(n_down)]
        blocks = [conv(nc, widths[0], use_bn=False)]
        for i in range(1, n_down):
            blocks.append(conv(widths[i - 1], widths[i], use_bn=True))
        self.features = nn.Sequential(*blocks)
        last_ch = widths[-1]
        self.head = nn.Conv2d(last_ch, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.sigmoid(self.head(h))
        return out.view(out.size(0), 1).squeeze(1)


# --------------------------------------------------------------------------- #
# 便捷构造函数：从 checkpoint 里存的 hparams 还原模型
# --------------------------------------------------------------------------- #


def build_generator(hparams: dict) -> Generator:
    return Generator(
        z_dim=int(hparams.get("z_dim", 100)),
        ngf=int(hparams.get("ngf", 64)),
        image_size=int(hparams.get("image_size", 64)),
    )


def build_discriminator(hparams: dict) -> Discriminator:
    return Discriminator(
        ndf=int(hparams.get("ndf", 64)),
        image_size=int(hparams.get("image_size", 64)),
        spectral_norm=bool(hparams.get("spectral_norm", False)),
        sigmoid=bool(hparams.get("d_sigmoid", False)),
    )
