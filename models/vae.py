"""卷积变分自编码器（ConvVAE / beta-VAE）。

编码器把 64x64 人脸压到 latent_dim 维的高斯分布，解码器再重建回图像。
与 GAN 的对比是这次实验的重点，所以这里特意把几处细节写清楚：

  * 重建损失：默认用 MSE（对应高斯似然），也可以换成 L1 或 BCE；
  * KL 项：逐维求和后按 batch 平均，beta 控制权衡；
  * 支持 KL warm-up（先重建、后加 KL），避免后验塌缩（posterior collapse）；
  * 前向返回 (recon, mu, logvar, z)，其中 z = mu + eps*std 只在训练时采样，
    评估重建质量时用 mu（确定性编码），结果更稳定。

约定：输入/输出都归一化到 [-1, 1]，最后一层用 Tanh。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.Module:
    return nn.BatchNorm2d(ch)


class ConvEncoder(nn.Module):
    def __init__(self, image_size: int = 64, nc: int = 3, base: int = 64, latent_dim: int = 128):
        super().__init__()
        if image_size < 8 or (image_size & (image_size - 1)) != 0:
            raise ValueError(f"image_size 必须是 >=8 的 2 的幂，收到 {image_size}")
        n_down = 0
        s = image_size
        while s > 4:
            s //= 2
            n_down += 1

        widths = [min(base * (2 ** i), base * 8) for i in range(n_down)]
        blocks = [nn.Sequential(
            nn.Conv2d(nc, widths[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )]
        for i in range(1, n_down):
            blocks.append(nn.Sequential(
                nn.Conv2d(widths[i - 1], widths[i], 4, 2, 1, bias=False),
                _norm(widths[i]),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.features = nn.Sequential(*blocks)
        self.last_ch = widths[-1]
        self.out_dim = self.last_ch * 4 * 4
        self.fc_mu = nn.Linear(self.out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.out_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, image_size: int = 64, nc: int = 3, base: int = 64, latent_dim: int = 128):
        super().__init__()
        n_up = 0
        s = 4
        while s < image_size:
            s *= 2
            n_up += 1

        widths = [min(base * (2 ** i), base * 8) for i in range(n_up)]
        widths = list(reversed(widths))  # 从宽到窄

        self.fc = nn.Linear(latent_dim, widths[0] * 4 * 4)
        blocks = []
        in_ch = widths[0]
        for i in range(1, len(widths)):
            out_ch = widths[i]
            blocks += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                _norm(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        blocks += [
            nn.ConvTranspose2d(in_ch, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*blocks)
        self.start_ch = widths[0]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.start_ch, 4, 4)
        return self.net(h)


class ConvVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        nc: int = 3,
        base: int = 64,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = ConvEncoder(image_size, nc, base, latent_dim)
        self.decoder = ConvDecoder(image_size, nc, base, latent_dim)

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """评估用：用 mu 直接解码（确定性，不含采样噪声）。"""
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """从先验 N(0, I) 采样，生成 n 张图。"""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    # ------------------------------------------------------------------ #
    @staticmethod
    def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I))，返回每个样本的标量 [B]。"""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.flatten(1).sum(dim=1)

    @staticmethod
    def recon_loss_per_sample(
        recon: torch.Tensor, target: torch.Tensor, kind: str = "mse"
    ) -> torch.Tensor:
        """逐样本重建损失（对像素求平均后乘以通道数），返回 [B]。"""
        if kind == "mse":
            return (recon - target).pow(2).flatten(1).mean(dim=1) * recon.size(1)
        if kind == "l1":
            return (recon - target).abs().flatten(1).mean(dim=1) * recon.size(1)
        if kind == "bce":
            # 把 [-1,1] 映到 (0,1) 再算二元交叉熵
            r = (recon + 1) * 0.5
            t = (target + 1) * 0.5
            return F.binary_cross_entropy(r.clamp(1e-6, 1 - 1e-6), t, reduction="none") \
                .flatten(1).sum(dim=1)
        raise ValueError(f"未知重建损失: {kind}")


def build_vae(hparams: dict) -> ConvVAE:
    return ConvVAE(
        image_size=int(hparams.get("image_size", 64)),
        base=int(hparams.get("base", 64)),
        latent_dim=int(hparams.get("latent_dim", 128)),
    )
