"""图像质量指标：PSNR、SSIM（纯 PyTorch 实现，不依赖 skimage）。

约定
----
* 输入张量形状 [B, C, H, W]，取值范围 [0, 1]（用 to01() 从 [-1,1] 转过来）。
* 返回的是 **逐图** 的指标张量 [B]，由调用方再求 mean/std。
* SSIM 采用高斯窗（window=11, sigma=1.5）、逐通道计算后取平均、
  使用 valid 区域（不做边界填充），与 skimage 的
  ``structural_similarity(..., gaussian_weights=True, sigma=1.5,
  use_sample_covariance=False)`` 结果基本一致。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

__all__ = [
    "psnr",
    "ssim",
    "psnr_ssim",
    "summarize",
    "nearest_neighbor_pairs",
    "nearest_neighbor_metrics",
]


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """逐图 PSNR(dB)。pred/target: [B,C,H,W]，取值 [0,1]。"""
    if pred.shape != target.shape:
        raise ValueError(f"形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
    mse = (pred - target).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10((data_range ** 2) / mse)


# --------------------------------------------------------------------------- #
# SSIM
# --------------------------------------------------------------------------- #


def _gaussian_window(window: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.expand(channels, 1, window, window).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """逐图 SSIM。pred/target: [B,C,H,W]，取值 [0,1]，返回 [B]。"""
    if pred.shape != target.shape:
        raise ValueError(f"形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
    b, c, h, w = pred.shape
    if h < window or w < window:
        # 图像太小时退化为全局统计，避免 valid 卷积为空
        window = min(h, w) if min(h, w) % 2 == 1 else max(1, min(h, w) - 1)

    kernel = _gaussian_window(window, sigma, c, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, kernel, groups=c)
    mu_y = F.conv2d(target, kernel, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den.clamp_min(1e-12)

    # 逐通道求均值（与 skimage 对 RGB 的处理一致），再对空间求均值
    return ssim_map.mean(dim=(1, 2, 3))


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """一次算两个指标，返回 {"psnr": [B], "ssim": [B]}。"""
    return {"psnr": psnr(pred, target), "ssim": ssim(pred, target)}


def summarize(values) -> Dict[str, float]:
    """把逐图指标列表/张量汇总成 mean/std/min/max/n。"""
    if isinstance(values, torch.Tensor):
        v = values.detach().float().flatten().cpu()
    else:
        v = torch.as_tensor(list(values), dtype=torch.float32)
    if v.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.numel()),
    }


# --------------------------------------------------------------------------- #
# 生成样本 ↔ 真实图像的最近邻配对
# --------------------------------------------------------------------------- #
#
# 说明：GAN/VAE 都是无条件生成模型，采样得到的图与测试集真实图之间
# **没有一一对应关系**，严格意义上不能直接算 PSNR/SSIM。
# 为了给出一个可比较的参考值，这里采用"最近邻配对"协议：
#   对每张生成图，在真实图库中找像素空间（16x16 缩略图上算 L2）最近的一张，
#   把这对图当作 (pred, GT) 计算 PSNR/SSIM。
# 这个数值只反映"生成分布是否覆盖到真实样本附近"，不应当与
# 重建类指标（VAE 重建、GAN 反演重建）混为一谈。


@torch.no_grad()
def nearest_neighbor_pairs(
    generated: torch.Tensor,
    real: torch.Tensor,
    thumb: int = 16,
    chunk: int = 128,
) -> torch.Tensor:
    """返回每张生成图对应的真实图索引 [N]。generated/real: [N|M,3,S,S] in [0,1]。"""
    if generated.numel() == 0 or real.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    g = F.adaptive_avg_pool2d(generated.float(), thumb).flatten(1)
    r = F.adaptive_avg_pool2d(real.float(), thumb).flatten(1)
    r_norm = r.pow(2).sum(dim=1)  # [M]

    idx = []
    for start in range(0, g.shape[0], chunk):
        gc = g[start:start + chunk]  # [b, D]
        # ||g||^2 - 2 g·r^T + ||r||^2
        d = gc.pow(2).sum(dim=1, keepdim=True) - 2.0 * gc @ r.t() + r_norm.unsqueeze(0)
        idx.append(d.argmin(dim=1))
    return torch.cat(idx, dim=0)


@torch.no_grad()
def nearest_neighbor_metrics(
    generated: torch.Tensor,
    real: torch.Tensor,
    thumb: int = 16,
    chunk: int = 128,
) -> Dict[str, torch.Tensor]:
    """最近邻配对后的 PSNR/SSIM，返回 {"psnr":[N], "ssim":[N], "index":[N]}。"""
    idx = nearest_neighbor_pairs(generated, real, thumb=thumb, chunk=chunk)
    paired = real[idx.to(real.device)]
    out = psnr_ssim(generated, paired)
    out["index"] = idx
    return out
