"""FID（Fréchet Inception Distance）的轻量实现。

不依赖 pytorch-fid，直接用 torchvision 的 Inception-v3（ImageNet 预训练权重）：
  1. 把 [0,1] 的图像 resize 到 299x299，用 ImageNet 的 mean/std 归一化；
  2. 取 2048 维的 pool 特征（去掉最后的 fc）；
  3. 对两组特征分别估计均值 mu 与协方差 Sigma，
     FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})。

注意
----
* 第一次运行会联网下载 inception_v3 权重（~100MB）；服务器无外网时请预先下载，
  或用 --skip-fid 跳过。
* FID 是分布级指标，样本量越大越稳（>= 2000 张比较有意义）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inception_features(device: torch.device):
    from torchvision import models

    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, transform_input=False, aux_logits=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(images: torch.Tensor, batch: int = 50) -> np.ndarray:
        feats = []
        for i in range(0, images.size(0), batch):
            x = images[i:i + batch].to(device).float()
            if x.size(-1) != 299:
                x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            f = model(x)
            if isinstance(f, tuple):
                f = f[0]
            feats.append(f.reshape(f.size(0), -1).cpu())
        return torch.cat(feats, 0).numpy().astype(np.float64)

    return extract


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """对称半正定矩阵的平方根（特征值分解，比 scipy.linalg.sqrtm 更稳）。"""
    mat = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def compute_fid_from_features(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    mu_r, mu_g = f_real.mean(0), f_fake.mean(0)
    sigma_r = np.cov(f_real, rowvar=False)
    sigma_g = np.cov(f_fake, rowvar=False)
    # Tr((Sigma_r Sigma_g)^{1/2}) 用对称化后的矩阵开方再取迹，避免复数值
    cov_mean = _sqrtm_psd(sigma_r @ sigma_g)
    diff = mu_r - mu_g
    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(cov_mean))


@torch.no_grad()
def fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | str = "cpu",
    batch: int = 50,
) -> float:
    """real/fake: [N,3,H,W] in [0,1]，建议 N>=1000。"""
    device = torch.device(device)
    extract = _inception_features(device)
    f_real = extract(real_images, batch)
    f_fake = extract(fake_images, batch)
    return compute_fid_from_features(f_real, f_fake)


if __name__ == "__main__":  # 自测：同分布应该得到接近 0 的值
    torch.manual_seed(0)
    a = torch.rand(16, 3, 64, 64)
    b = torch.rand(16, 3, 64, 64)
    print("FID(random vs random) =", fid(a, b, device="cpu", batch=8))
