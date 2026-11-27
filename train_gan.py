"""DCGAN 训练脚本（CelebA 64x64）。

用法
----
    python train_gan.py --data data/celeba64 --out runs/gan64 --epochs 30 --batch-size 128 --amp

要点
----
* 每步先更新 D 再更新 G（也可以改 n_critic>1，代码里已支持）；
* G 的损失用 non-saturating 形式（-log D(G(z))），比 minimax 稳定；
* 生成器维护一份 EMA 权重，采样时默认用 EMA（更平滑）；
* 每个 epoch 存一次样本网格 + checkpoint，并记录 loss 到 csv。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import build_loader
from models.gan import Discriminator, Generator
from utils import (
    AverageMeter,
    autocast,
    CSVLogger,
    ModelEMA,
    Timer,
    add_common_args,
    describe_device,
    get_device,
    human_time,
    load_checkpoint,
    make_grad_scaler,
    save_checkpoint,
    save_grid,
    save_json,
    set_seed,
    setup_cudnn,
)


def parse_args():
    ap = argparse.ArgumentParser(description="DCGAN on CelebA")
    add_common_args(ap)
    ap.add_argument("--z-dim", type=int, default=100)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=0, help="0 表示从 meta.json 读取")
    ap.add_argument("--n-critic", type=int, default=1, help="每更新 1 次 G，更新几次 D")
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--spectral-norm", action="store_true", help="判别器用谱归一化（更稳）")
    ap.add_argument("--loss", type=str, default="bce", choices=["bce", "hinge", "lsgan"])
    ap.add_argument("--ema-decay", type=float, default=0.999, help="0 表示关闭 EMA")
    ap.add_argument("--save-every", type=int, default=1, help="每多少个 epoch 存 checkpoint")
    ap.add_argument("--sample-n", type=int, default=64, help="每个 epoch 保存多少张生成样本")
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #


def d_loss_fn(kind: str, real_logit: torch.Tensor, fake_logit: torch.Tensor):
    if kind == "bce":
        real = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
        fake = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        return real + fake, real.detach(), fake.detach()
    if kind == "hinge":
        real = F.relu(1.0 - real_logit).mean()
        fake = F.relu(1.0 + fake_logit).mean()
        return real + fake, real.detach(), fake.detach()
    if kind == "lsgan":
        real = 0.5 * (real_logit - 1.0).pow(2).mean()
        fake = 0.5 * fake_logit.pow(2).mean()
        return real + fake, real.detach(), fake.detach()
    raise ValueError(kind)


def g_loss_fn(kind: str, fake_logit: torch.Tensor) -> torch.Tensor:
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
    if kind == "hinge":
        return -fake_logit.mean()
    if kind == "lsgan":
        return 0.5 * (fake_logit - 1.0).pow(2).mean()
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_cudnn()
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    # 数据
    train_loader, train_ds = build_loader(
        args.data, "train", batch_size=args.batch_size, num_workers=args.num_workers, augment=True
    )
    image_size = args.image_size or train_ds.size
    steps_per_epoch = len(train_loader)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    print("=" * 72)
    print(f"DCGAN  |  device={describe_device(device)}")
    print(f"data={args.data}  train={len(train_ds)}  image_size={image_size}")
    print(f"epochs={args.epochs}  batch={args.batch_size}  steps/epoch={steps_per_epoch}  "
          f"total_steps={steps_per_epoch * args.epochs}")
    print("=" * 72)

    # 模型
    G = Generator(z_dim=args.z_dim, ngf=args.ngf, image_size=image_size).to(device)
    D = Discriminator(ndf=args.ndf, image_size=image_size,
                      spectral_norm=args.spectral_norm).to(device)

    hparams = {
        "model": "dcgan",
        "z_dim": args.z_dim,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "image_size": image_size,
        "spectral_norm": bool(args.spectral_norm),
        "loss": args.loss,
        "ema_decay": args.ema_decay,
        "data": str(args.data),
    }
    save_json(hparams, out_dir / "hparams.json")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler_g = make_grad_scaler(use_amp)
    scaler_d = make_grad_scaler(use_amp)

    ema = ModelEMA(G, decay=args.ema_decay) if args.ema_decay > 0 else None
    # 采样用的固定噪声，保证每个 epoch 的对比图是"同一批 z"
    fixed_z = torch.randn(args.sample_n, args.z_dim, device=device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        print(f"[resume] 从 epoch {start_epoch} / step {global_step} 继续")

    csv_log = CSVLogger(
        out_dir / "loss.csv",
        ["epoch", "step", "loss_d", "loss_g", "d_real", "d_fake", "d_acc_real", "d_acc_fake", "lr"],
    )
    timer = Timer()
    best = {"step": 0, "loss_g": float("inf")}

    @torch.no_grad()
    def sample_and_save(epoch: int, tag: str = "") -> None:
        """用 EMA（或原始）权重采样并保存网格图，z 固定，便于观察逐轮变化。"""
        G.eval()
        if ema is not None:
            ema.copy_to(sampler)
        else:
            sampler.load_state_dict(G.state_dict())
        imgs = sampler(fixed_z)
        name = f"epoch_{epoch:03d}{tag}.png"
        save_grid(imgs, out_dir / "samples" / name, nrow=8)
        G.train()

    # 采样用的独立模型（不参与训练），避免每轮都新建网络
    sampler = Generator(z_dim=args.z_dim, ngf=args.ngf, image_size=image_size).to(device)
    for p in sampler.parameters():
        p.requires_grad_(False)

    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        m_d, m_g = AverageMeter("d"), AverageMeter("g")
        m_dr, m_df = AverageMeter("d_real"), AverageMeter("d_fake")
        acc_r = AverageMeter("acc_real")
        acc_f = AverageMeter("acc_fake")
        t_epoch = time.time()

        for step, (real, _) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # ---------------- 更新 D ----------------
            for _ in range(max(1, args.n_critic)):
                z = torch.randn(b, args.z_dim, device=device)
                with torch.no_grad():
                    fake = G(z)  # D 步不需要 G 的计算图，省显存
                with autocast(scaler_d.is_enabled()):
                    real_logit = D(real)
                    fake_logit = D(fake)
                    loss_d, lr_real, lr_fake = d_loss_fn(args.loss, real_logit, fake_logit)
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

            # ---------------- 更新 G ----------------
            z = torch.randn(b, args.z_dim, device=device)
            with autocast(scaler_g.is_enabled()):
                fake = G(z)
                fake_logit = D(fake)
                loss_g = g_loss_fn(args.loss, fake_logit)
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            if ema is not None:
                ema.update(G)

            global_step += 1
            m_d.update(loss_d.item(), b)
            m_g.update(loss_g.item(), b)
            m_dr.update(float(lr_real), b)
            m_df.update(float(lr_fake), b)
            acc_r.update(float((real_logit.detach() > 0).float().mean()), b)
            acc_f.update(float((fake_logit.detach() > 0).float().mean()), b)

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                print(f"[{epoch:03d}][{step + 1:04d}/{steps_per_epoch}] "
                      f"D={m_d.avg:.3f} G={m_g.avg:.3f} "
                      f"D(x)={m_dr.avg:+.2f} D(G(z))={m_df.avg:+.2f} "
                      f"acc_r={acc_r.avg:.2f} acc_f={acc_f.avg:.2f} "
                      f"{human_time(time.time() - t_epoch)}")

        epoch_time = time.time() - t_epoch
        lr_now = opt_g.param_groups[0]["lr"]
        csv_log.log(epoch=epoch, step=global_step, loss_d=m_d.avg, loss_g=m_g.avg,
                    d_real=m_dr.avg, d_fake=m_df.avg,
                    d_acc_real=acc_r.avg, d_acc_fake=acc_f.avg, lr=lr_now)
        print(f"== epoch {epoch:03d}  D={m_d.avg:.4f}  G={m_g.avg:.4f}  "
              f"D(x)={m_dr.avg:+.3f}  D(G(z))={m_df.avg:+.3f}  "
              f"用时 {human_time(epoch_time)}  累计 {timer}")

        sample_and_save(epoch)

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "hparams": hparams,
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        save_checkpoint(state, out_dir / "last.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(state, out_dir / f"epoch_{epoch:03d}.pt")

        # 用一个简单信号挑"最好"的权重：G 损失最低（仅作参考）
        if m_g.avg < best["loss_g"]:
            best = {"step": global_step, "loss_g": m_g.avg, "epoch": epoch}
            save_checkpoint(state, out_dir / "best.pt")

    csv_log.close()
    save_json({**hparams, "best": best, "final_epoch": args.epochs - 1,
               "train_seconds": timer.elapsed()}, out_dir / "train_summary.json")
    print(f"训练结束，权重保存在 {out_dir}（last.pt / best.pt / epoch_*.pt）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 已保存 last.pt，可用 --resume 继续训练")
        sys.exit(130)
