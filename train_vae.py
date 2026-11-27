"""ConvVAE 训练脚本（CelebA 64x64）。

损失 = 重建项 + beta * KL
  * 重建项默认 MSE（等价于固定方差的高斯解码器）；
  * beta 可调：beta=1 是标准 VAE，beta>1 更强调解耦（beta-VAE），
    beta<1 重建更清晰；
  * --kl-warmup N：前 N 个 epoch 把 beta 从 0 线性升到目标值，
    能有效缓解后验塌缩（KL 直接归零、latent 失效）。

用法
----
    python train_vae.py --data data/celeba64 --out runs/vae64 --epochs 30 \
        --latent-dim 128 --beta 1.0 --kl-warmup 5 --amp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from dataset import build_loader
from models.vae import ConvVAE
from utils import (
    AverageMeter,
    autocast,
    CSVLogger,
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
    to01,
)


def parse_args():
    ap = argparse.ArgumentParser(description="ConvVAE on CelebA")
    add_common_args(ap)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--base", type=int, default=64, help="编码器/解码器基础通道数")
    ap.add_argument("--image-size", type=int, default=0, help="0 表示从 meta.json 读取")
    ap.add_argument("--beta", type=float, default=1.0, help="KL 项权重")
    ap.add_argument("--kl-warmup", type=int, default=0, help="前 N 个 epoch 线性 warm-up beta")
    ap.add_argument("--recon", type=str, default="mse", choices=["mse", "l1", "bce"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--sample-n", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def beta_at(epoch: int, target: float, warmup: int) -> float:
    if warmup <= 0:
        return target
    return target * min(1.0, (epoch + 1) / float(warmup))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_cudnn()
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    train_loader, train_ds = build_loader(
        args.data, "train", batch_size=args.batch_size, num_workers=args.num_workers, augment=True
    )
    image_size = args.image_size or train_ds.size
    steps_per_epoch = len(train_loader)
    if args.max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    print("=" * 72)
    print(f"ConvVAE  |  device={describe_device(device)}")
    print(f"data={args.data}  train={len(train_ds)}  image_size={image_size}")
    print(f"epochs={args.epochs}  batch={args.batch_size}  latent={args.latent_dim}  "
          f"beta={args.beta}  warmup={args.kl_warmup}")
    print("=" * 72)

    model = ConvVAE(image_size=image_size, base=args.base, latent_dim=args.latent_dim).to(device)
    hparams = {
        "model": "convvae",
        "image_size": image_size,
        "base": args.base,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "recon": args.recon,
        "data": str(args.data),
    }
    save_json(hparams, out_dir / "hparams.json")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] 从 epoch {start_epoch} 继续")

    fixed_z = torch.randn(args.sample_n, args.latent_dim, device=device)
    csv_log = CSVLogger(
        out_dir / "loss.csv",
        ["epoch", "step", "loss", "recon", "kl", "beta", "psnr_epoch", "lr"],
    )
    timer = Timer()
    best = {"epoch": -1, "loss": float("inf")}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        beta = beta_at(epoch, args.beta, args.kl_warmup)
        m_tot, m_rec, m_kl = AverageMeter("loss"), AverageMeter("recon"), AverageMeter("kl")
        t_epoch = time.time()

        for step, (real, _) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            real = real.to(device, non_blocking=True)
            with autocast(scaler.is_enabled()):
                recon, mu, logvar, _ = model(real, sample=True)
                rec = model.recon_loss_per_sample(recon, real, args.recon).mean()
                kl = model.kl_per_sample(mu, logvar).mean()
                loss = rec + beta * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # VAE 偶尔会有梯度尖峰
            scaler.step(opt)
            scaler.update()

            m_tot.update(loss.item(), real.size(0))
            m_rec.update(rec.item(), real.size(0))
            m_kl.update(kl.item(), real.size(0))

            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                print(f"[{epoch:03d}][{step + 1:04d}/{steps_per_epoch}] "
                      f"loss={m_tot.avg:.2f} recon={m_rec.avg:.2f} kl={m_kl.avg:.1f} "
                      f"beta={beta:.2f} {human_time(time.time() - t_epoch)}")

        scheduler.step()

        # ---- epoch 评估：重建 PSNR（用 mu 确定性编码）----
        model.eval()
        psnrs = []
        with torch.no_grad():
            for i, (real, _) in enumerate(train_loader):
                if i >= 10:  # 只看 10 个 batch，够用且快
                    break
                real = real.to(device, non_blocking=True)
                rec = model.reconstruct(real)
                mse = (to01(rec) - to01(real)).pow(2).flatten(1).mean(dim=1)
                psnrs.append((10 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean().item())
        psnr_epoch = sum(psnrs) / max(len(psnrs), 1)

        lr_now = opt.param_groups[0]["lr"]
        csv_log.log(epoch=epoch, step=steps_per_epoch, loss=m_tot.avg, recon=m_rec.avg,
                    kl=m_kl.avg, beta=beta, psnr_epoch=psnr_epoch, lr=lr_now)
        print(f"== epoch {epoch:03d}  loss={m_tot.avg:.2f}  recon={m_rec.avg:.2f}  "
              f"kl={m_kl.avg:.1f}  beta={beta:.2f}  recon_PSNR={psnr_epoch:.2f}dB  "
              f"用时 {human_time(time.time() - t_epoch)}  累计 {timer}")

        # ---- 存样本：先验采样 + 训练集前 8 张的重建 ----
        with torch.no_grad():
            prior = model.decode(fixed_z)
            save_grid(prior, out_dir / "samples" / f"prior_{epoch:03d}.png", nrow=8)
            real8, _ = next(iter(train_loader))
            real8 = real8[:8].to(device)
            rec8 = model.reconstruct(real8)
            # 上排真实、下排重建，方便直观对比
            pair = torch.cat([real8, rec8], dim=0)
            save_grid(pair, out_dir / "samples" / f"recon_{epoch:03d}.png", nrow=8)

        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                 "hparams": hparams}
        save_checkpoint(state, out_dir / "last.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(state, out_dir / f"epoch_{epoch:03d}.pt")
        if m_tot.avg < best["loss"]:
            best = {"epoch": epoch, "loss": m_tot.avg, "psnr": psnr_epoch}
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
