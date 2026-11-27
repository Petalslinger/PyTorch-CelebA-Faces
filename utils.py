"""通用工具：随机种子、设备、日志、权重滑动平均(EMA)、图像网格保存。

所有脚本共享这里的函数，避免重复代码。
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# 随机性与设备
# --------------------------------------------------------------------------- #


def set_seed(seed: int = 0) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_device(arg: str = "auto") -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return f"cuda:{idx} ({name}, {mem:.1f} GB)"
    return str(device)


def setup_cudnn(benchmark: bool = True) -> None:
    """固定尺寸输入下打开 cudnn benchmark 可以提速。"""
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = not benchmark


# --------------------------------------------------------------------------- #
# 数值 / 图像
# --------------------------------------------------------------------------- #


def to01(x: torch.Tensor) -> torch.Tensor:
    """把 [-1, 1] 的生成结果映射到 [0, 1]，便于保存图片和算指标。"""
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_grid(tensor: torch.Tensor, path: str | Path, nrow: int = 8, padding: int = 2) -> None:
    """tensor 取值 [-1, 1] 或 [0, 1]，自动判断后保存成网格图。"""
    from torchvision.utils import save_image

    t = tensor.detach().float().cpu()
    if t.min() < -0.05:  # 大概率是 [-1, 1]
        t = to01(t)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(t, str(path), nrow=nrow, padding=padding)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #


class AverageMeter:
    """滑动窗口平均，用于打印 loss。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}={self.avg:.4f}"


class CSVLogger:
    """把每个 step / epoch 的指标追加写入 csv，方便后面画 loss 曲线。"""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # 原子写入，避免训练中断损坏权重


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=device, weights_only=False)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """兼容 DataParallel / DDP 保存的 state_dict。"""
    if all(not k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


class ModelEMA:
    """生成器权重的指数滑动平均，通常能明显提升采样质量。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # 浮点参数用 float32 影子副本；int 型 buffer（如 num_batches_tracked）保持原 dtype
        self.shadow: Dict[str, torch.Tensor] = {
            k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
            for k, v in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # 前期用 1/(n+1) 做 warmup，避免刚开始 EMA 被初始权重拖住
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """把影子权重复制进 model（原地 copy，保持 dtype/device 正确）。"""
        target = model.state_dict()
        for k, v in target.items():
            v.copy_(self.shadow[k].to(device=v.device, dtype=v.dtype))

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def __str__(self) -> str:  # pragma: no cover
        return human_time(self.elapsed())


def make_optimizer(model: nn.Module, lr: float, betas: Optional[tuple] = None, wd: float = 0.0):
    betas = betas or (0.5, 0.999)
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)


# --------------------------------------------------------------------------- #
# 混合精度：兼容 torch 2.0 ~ 2.5（torch.amp.GradScaler 是 2.3 才有的）
# --------------------------------------------------------------------------- #


def make_grad_scaler(enabled: bool = True):
    if enabled and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    if hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return torch.amp.GradScaler(enabled=enabled)


def autocast(enabled: bool = True):
    if hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=enabled)


def add_common_args(parser) -> None:
    """训练脚本共用的命令行参数。"""
    parser.add_argument("--data", type=str, required=True, help="prepare_data.py 生成的目录，如 data/celeba64")
    parser.add_argument("--out", type=str, required=True, help="输出目录，如 runs/gan64")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--amp", action="store_true", help="开启混合精度（A100/3090/4060 等建议开）")
    parser.add_argument("--resume", type=str, default="", help="断点续训的 checkpoint 路径")
    parser.add_argument("--max-steps", type=int, default=0, help=">0 时每个 epoch 只跑这么多 step（调试用）")
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "test"])
