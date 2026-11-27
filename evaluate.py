"""评估脚本：定量（PSNR / SSIM / 可选 FID）+ 定性（生成图 vs GT 对比图）+ 自动生成结论。

为什么 GAN 与 VAE 的评估方式不一样？
------------------------------------------------
* VAE 是**自编码器**：给一张真实图 x，它输出重建 x_hat，两者天然一一对应，
  可以直接算 PSNR(x_hat, x) / SSIM(x_hat, x)，GT 就是输入图。
* GAN 是**无条件生成器**：从随机噪声 z 生成图像，输出与任何真实图都没有对应关系，
  严格来说不能直接算 PSNR/SSIM。本脚本给出两种补充方案：
    1) 最近邻配对（默认）：每张生成图在测试集里找最像的真实图作为"伪 GT"，
       衡量生成分布是否落在真实数据附近；
    2) 潜在反演（--inversion-steps > 0）：为每张真实图优化出一个 z，
       使 G(z) 尽量接近该图，于是得到与 VAE 同口径的重建指标。
       反演重建反映的是"生成器能覆盖到多少真实图像"，不等于生成质量。

用法
----
    python evaluate.py --gan runs/gan64 --vae runs/vae64 --data data/celeba64 \
        --out runs/eval --n-samples 2048 --inversion-n 64 --inversion-steps 300 --fid
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import build_loader, read_meta
from metrics import nearest_neighbor_metrics, psnr, ssim, summarize
from models.gan import Generator
from models.vae import ConvVAE
from utils import (
    Timer,
    describe_device,
    get_device,
    human_time,
    load_checkpoint,
    load_json,
    save_grid,
    save_json,
    set_seed,
    setup_cudnn,
    to01,
)


# --------------------------------------------------------------------------- #
# 读取 checkpoint
# --------------------------------------------------------------------------- #


def _load_hparams(run_dir: Path, ckpt: dict) -> dict:
    if "hparams" in ckpt and ckpt["hparams"]:
        return dict(ckpt["hparams"])
    p = run_dir / "hparams.json"
    if p.exists():
        return load_json(p)
    raise FileNotFoundError(f"{run_dir} 里找不到 hparams（hparams.json 或 checkpoint['hparams']）")


def load_gan(run_dir: str | Path, device: torch.device, prefer: str = "ema"):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 不存在，请先训练 GAN")
    ckpt = load_checkpoint(ckpt_path, device)
    hp = _load_hparams(run_dir, ckpt)

    G = Generator(z_dim=hp["z_dim"], ngf=hp["ngf"], image_size=hp["image_size"])
    if prefer == "ema" and "ema" in ckpt:
        # EMA 影子权重是 float32，按 dtype/device 拷回，避免 state_dict 类型不匹配
        G.load_state_dict(ckpt["G"])
        shadow = ckpt["ema"]["shadow"]
        for k, v in G.state_dict().items():
            v.copy_(shadow[k].to(device=v.device, dtype=v.dtype))
        weight_tag = "ema"
    else:
        G.load_state_dict(ckpt["G"])
        weight_tag = "raw"
    G.eval().to(device)
    return G, hp, weight_tag


def load_vae(run_dir: str | Path, device: torch.device):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 不存在，请先训练 VAE")
    ckpt = load_checkpoint(ckpt_path, device)
    hp = _load_hparams(run_dir, ckpt)
    model = ConvVAE(image_size=hp["image_size"], base=hp["base"], latent_dim=hp["latent_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, hp


# --------------------------------------------------------------------------- #
# 生成 / 反演
# --------------------------------------------------------------------------- #


@torch.no_grad()
def sample_gan(G: Generator, n: int, device: torch.device, batch: int = 128) -> torch.Tensor:
    outs = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        z = torch.randn(m, G.z_dim, device=device)
        outs.append(G(z).cpu())
    return to01(torch.cat(outs, 0))


@torch.no_grad()
def sample_vae_prior(model: ConvVAE, n: int, device: torch.device, batch: int = 128) -> torch.Tensor:
    outs = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        outs.append(model.sample(m, device=device).cpu())
    return to01(torch.cat(outs, 0))


@torch.no_grad()
def reconstruct_vae(model: ConvVAE, images: torch.Tensor, device: torch.device,
                    batch: int = 128) -> torch.Tensor:
    outs = []
    for i in range(0, images.size(0), batch):
        x = images[i:i + batch].to(device)
        outs.append(model.reconstruct(x).cpu())
    return to01(torch.cat(outs, 0))


def invert_gan(
    G: Generator,
    targets: torch.Tensor,
    device: torch.device,
    steps: int = 300,
    lr: float = 0.05,
    reg: float = 1e-3,
) -> torch.Tensor:
    """为每张目标图优化一个 z，使 to01(G(z)) ≈ target。返回重建图像（CPU, [0,1]）。

    reg 是 z 的 L2 正则，防止优化到极端区域（那些区域生成质量通常很差）。
    """
    targets = targets.to(device).float()
    z = torch.zeros(targets.size(0), G.z_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        rec = to01(G(z))
        loss = F.mse_loss(rec, to01(targets)) + reg * z.pow(2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec = to01(G(z))
    return rec.cpu()


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #


def save_pair_grid(gt: torch.Tensor, pred: torch.Tensor, path: Path, title: str = "") -> None:
    """上排 GT、下排预测，一眼看出差距。"""
    n = min(gt.size(0), pred.size(0), 8)
    stacked = torch.cat([gt[:n], pred[:n]], dim=0)
    save_grid(stacked, path, nrow=n)
    print(f"      [图] {path.name}  {title}")


def save_bar_chart(summary: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = [m for m in ("VAE-recon", "GAN-inversion", "GAN-NN") if m in summary]
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")):
        vals = [summary[m][key]["mean"] for m in models]
        errs = [summary[m][key]["std"] for m in models]
        bars = ax.bar(models, vals, yerr=errs, capsize=4,
                      color=["#4C78A8", "#F58518", "#B279A2"][:len(models)])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("GAN vs VAE: PSNR / SSIM (higher is better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      [图] {path.name}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser(description="GAN / VAE 评估")
    ap.add_argument("--gan", type=str, default="", help="GAN 训练输出目录（含 last.pt）")
    ap.add_argument("--vae", type=str, default="", help="VAE 训练输出目录（含 last.pt）")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=2048, help="生成样本数（算指标用）")
    ap.add_argument("--n-gallery", type=int, default=4096, help="最近邻匹配用的真实图库大小")
    ap.add_argument("--inversion-n", type=int, default=64, help="GAN 反演的真实图数量")
    ap.add_argument("--inversion-steps", type=int, default=300, help="每个 z 的优化步数")
    ap.add_argument("--inversion-lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fid", action="store_true", help="额外计算 FID（需要联网下载 inception 权重）")
    ap.add_argument("--fid-n", type=int, default=2000)
    ap.add_argument("--prefer", type=str, default="ema", choices=["ema", "raw"],
                    help="GAN 采样用 EMA 权重还是原始权重")
    return ap.parse_args()


def main() -> None:
    """注意：这里不能加 @torch.no_grad()——GAN 潜在反演需要对 z 求梯度。
    采样/重建/指标这些子函数各自带 @torch.no_grad() 装饰器。"""
    args = parse_args()
    if not args.gan and not args.vae:
        raise SystemExit("至少指定 --gan 或 --vae 之一")
    set_seed(args.seed)
    setup_cudnn()
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    meta = read_meta(args.data)
    print("=" * 72)
    print(f"评估  |  device={describe_device(device)}")
    print(f"data={args.data}  split={args.split}  meta={meta.get('counts', {})}")
    print("=" * 72)

    loader, ds = build_loader(
        args.data, args.split, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
    )
    # 取一批真实图作为 GT（batch 里第二项是索引，这里只需要图像）
    n_gallery = min(args.n_gallery, len(ds))
    gal_loader, _ = build_loader(
        args.data, args.split, batch_size=n_gallery, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
    )
    real_all = next(iter(gal_loader))[0].cpu()
    print(f"真实图库: {real_all.size(0)} 张（{args.split} split 共 {len(ds)} 张）")
    gt_sub = real_all[: args.inversion_n]

    results: dict = {"config": vars(args), "meta": meta, "metrics": {}}
    summary: dict = {}

    # ---------------- VAE ----------------
    if args.vae:
        print("\n[VAE] 加载模型并计算重建指标 ...")
        vae, vhp = load_vae(args.vae, device)
        t0 = time.time()
        rec = reconstruct_vae(vae, real_all, device, args.batch_size)
        gt = to01(real_all)
        v_psnr = psnr(rec, gt)
        v_ssim = ssim(rec, gt)
        summary["VAE-recon"] = {"psnr": summarize(v_psnr), "ssim": summarize(v_ssim)}
        print(f"      VAE 重建: PSNR={v_psnr.mean():.2f}dB  SSIM={v_ssim.mean():.4f}  "
              f"({real_all.size(0)} 张, {human_time(time.time() - t0)})")
        save_pair_grid(gt, rec, out_dir / "compare_vae_recon.png", "VAE 重建(下) vs GT(上)")
        save_grid(rec[:64], out_dir / "samples_vae_recon.png", nrow=8)

        prior = sample_vae_prior(vae, min(args.n_samples, 64), device)
        save_grid(prior, out_dir / "samples_vae_prior.png", nrow=8)
        # 先验采样的"最近邻伪 GT"指标，与 GAN 用同一口径
        nn_v = nearest_neighbor_metrics(prior, gt)
        summary["VAE-prior-NN"] = {
            "psnr": summarize(nn_v["psnr"]), "ssim": summarize(nn_v["ssim"])
        }
        print(f"      VAE 先验采样(最近邻): PSNR={nn_v['psnr'].mean():.2f}dB  "
              f"SSIM={nn_v['ssim'].mean():.4f}")
        results["metrics"]["vae"] = {
            "recon_psnr": [round(float(x), 4) for x in v_psnr],
            "recon_ssim": [round(float(x), 4) for x in v_ssim],
            "prior_nn_psnr": [round(float(x), 4) for x in nn_v["psnr"]],
            "prior_nn_ssim": [round(float(x), 4) for x in nn_v["ssim"]],
            "hparams": vhp,
        }
        if args.fid:
            from fid import fid

            n = min(args.fid_n, real_all.size(0), prior.size(0))
            f = fid(to01(real_all[:n]), prior[:n], device=device)
            summary["VAE-prior-NN"]["fid"] = f
            print(f"      VAE FID = {f:.2f}")

    # ---------------- GAN ----------------
    if args.gan:
        print("\n[GAN] 加载模型并生成样本 ...")
        G, ghp, weight_tag = load_gan(args.gan, device, prefer=args.prefer)
        print(f"      使用权重: {weight_tag}")
        t0 = time.time()
        fake = sample_gan(G, args.n_samples, device, args.batch_size)
        print(f"      生成 {fake.size(0)} 张样本，用时 {human_time(time.time() - t0)}")
        save_grid(fake[:64], out_dir / "samples_gan.png", nrow=8)

        gt = to01(real_all)
        nn = nearest_neighbor_metrics(fake, gt)
        summary["GAN-NN"] = {"psnr": summarize(nn["psnr"]), "ssim": summarize(nn["ssim"])}
        print(f"      GAN 生成(最近邻): PSNR={nn['psnr'].mean():.2f}dB  "
              f"SSIM={nn['ssim'].mean():.4f}")
        # 生成图 / 最近邻 GT 对照
        idx = nn["index"]
        save_pair_grid(gt[idx[:8]], fake[:8], out_dir / "compare_gan_nn.png",
                       "GAN 生成(下) vs 最近邻 GT(上)")

        results["metrics"]["gan"] = {
            "nn_psnr": [round(float(x), 4) for x in nn["psnr"]],
            "nn_ssim": [round(float(x), 4) for x in nn["ssim"]],
            "nn_index": [int(i) for i in idx],
            "hparams": ghp,
            "weights": weight_tag,
        }

        if args.inversion_steps > 0 and args.inversion_n > 0:
            print(f"\n[GAN] 潜在反演：{args.inversion_n} 张图 x {args.inversion_steps} 步 ...")
            t0 = time.time()
            inv = invert_gan(G, gt_sub, device, steps=args.inversion_steps,
                             lr=args.inversion_lr)
            i_psnr = psnr(inv, gt_sub)
            i_ssim = ssim(inv, gt_sub)
            summary["GAN-inversion"] = {"psnr": summarize(i_psnr), "ssim": summarize(i_ssim)}
            print(f"      GAN 反演重建: PSNR={i_psnr.mean():.2f}dB  "
                  f"SSIM={i_ssim.mean():.4f}  用时 {human_time(time.time() - t0)}")
            save_pair_grid(gt_sub, inv, out_dir / "compare_gan_inversion.png",
                           "GAN 反演重建(下) vs GT(上)")
            save_grid(inv[:64], out_dir / "samples_gan_inversion.png", nrow=8)
            results["metrics"]["gan"]["inversion_psnr"] = [round(float(x), 4) for x in i_psnr]
            results["metrics"]["gan"]["inversion_ssim"] = [round(float(x), 4) for x in i_ssim]

        if args.fid:
            from fid import fid

            n = min(args.fid_n, real_all.size(0), fake.size(0))
            f = fid(to01(real_all[:n]), fake[:n], device=device)
            summary["GAN-NN"]["fid"] = f
            print(f"      GAN FID = {f:.2f}")

    # ---------------- 汇总 ----------------
    results["summary"] = summary
    results["elapsed_seconds"] = timer.elapsed()
    save_json(results, out_dir / "results.json")

    # 行优先打印一张表
    print("\n" + "=" * 72)
    print(f"{'方法':<18}{'PSNR(dB)':>16}{'SSIM':>18}{'FID':>10}")
    print("-" * 72)
    for name, m in summary.items():
        p, s = m["psnr"], m["ssim"]
        fid_s = f"{m['fid']:.2f}" if "fid" in m else "-"
        print(f"{name:<18}{p['mean']:>9.2f}±{p['std']:<6.2f}{s['mean']:>10.4f}±{s['std']:<6.4f}{fid_s:>10}")
    print("=" * 72)

    save_bar_chart(summary, out_dir / "metric_bar.png")
    write_report(out_dir, summary, args, meta)
    print(f"\n结果已保存: {out_dir / 'results.json'}")
    print(f"对比图: {out_dir}/compare_*.png  样本: {out_dir}/samples_*.png")
    print(f"自动结论: {out_dir / 'report.md'}")


def write_report(out_dir: Path, summary: dict, args, meta: dict) -> None:
    """根据本次指标自动生成一份中文结论草稿。

    所有结论都由真实计算结果驱动（比较大小后再选择措辞），
    不会出现"数字上落后却写成占优"的情况；但措辞仍属模板，
    实验报告里应结合定性图人工复核。
    """

    def mean(name: str, key: str = "psnr"):
        return summary[name][key]["mean"] if name in summary else None

    def fmt(name: str, key: str, nd: int = 2) -> str:
        if name not in summary:
            return "-"
        m = summary[name][key]
        return f"{m['mean']:.{nd}f} ± {m['std']:.{nd}f}"

    lines = [
        "# 实验结果：GAN 与 VAE 人脸生成对比",
        "",
        f"- 数据集：CelebA（{meta.get('image_size', '?')}x{meta.get('image_size', '?')}，"
        f"train/val/test = {meta.get('counts', {})}）",
        f"- 评估 split：{args.split}；生成样本数：{args.n_samples}；"
        f"最近邻图库：{args.n_gallery}；反演图数：{args.inversion_n}",
        "",
        "## 1. 定量结果",
        "",
        "| 方法 | PSNR (dB) | SSIM | 指标口径 |",
        "| --- | --- | --- | --- |",
    ]
    desc = {
        "VAE-recon": "重建：x̂ 与输入 x 一一对应，GT 真实",
        "GAN-inversion": "反演重建：为每张真实图优化 z 使 G(z)≈x",
        "GAN-NN": "随机生成：生成图匹配最近邻真实图作伪 GT",
        "VAE-prior-NN": "先验采样：同上的最近邻配对口径",
    }
    for name in ("VAE-recon", "GAN-inversion", "GAN-NN", "VAE-prior-NN"):
        if name in summary:
            lines.append(f"| {name} | {fmt(name, 'psnr')} | {fmt(name, 'ssim', 4)} | {desc[name]} |")

    lines += ["", "## 2. 结论（自动生成草稿，请结合定性图人工复核）", ""]

    vp, vs = mean("VAE-recon", "psnr"), mean("VAE-recon", "ssim")
    ip = mean("GAN-inversion", "psnr")
    gp, gs = mean("GAN-NN", "psnr"), mean("GAN-NN", "ssim")
    np_ = mean("VAE-prior-NN", "psnr")

    # (1) 重建口径对比：VAE 重建 vs GAN 反演（同口径）
    if vp is not None and ip is not None:
        if vp > ip:
            lines.append(
                f"- **同口径重建对比：VAE 更优**。VAE 重建 PSNR {vp:.2f}dB，"
                f"GAN 潜在反演重建 {ip:.2f}dB（差 {vp - ip:.2f}dB）。"
                f"VAE 一次前向即可重建，GAN 需要逐图优化 {args.inversion_steps} 步，"
                f"工程上 VAE 也更省。原因在于 VAE 的编码器显式学习了 x→z 的映射，"
                f"而 GAN 的潜空间没有可逆保证。"
            )
        else:
            lines.append(
                f"- **同口径重建对比：GAN 反演反而更高**（GAN {ip:.2f}dB vs VAE {vp:.2f}dB）。"
                f"这通常说明 VAE 的 latent 维度过小或 beta 过大导致重建被压缩，"
                f"可尝试增大 --latent-dim 或减小 --beta；同时也说明反演优化能「过拟合」到单张图。"
            )

    # (2) 分布覆盖：GAN 随机采样 vs VAE 先验采样（同为最近邻口径）
    if gp is not None and np_ is not None:
        better = "GAN" if gp > np_ else "VAE"
        lines.append(
            f"- **随机采样的分布覆盖（最近邻口径）**：GAN {gp:.2f}dB / SSIM {gs:.4f}，"
            f"VAE 先验采样 {np_:.2f}dB / SSIM {summary['VAE-prior-NN']['ssim']['mean']:.4f}，"
            f"数值上 {better} 更接近真实图库。该口径只衡量「生成图附近是否存在相似真实图」，"
            f"不衡量清晰度，必须配合 FID 与人眼判断。"
        )

    # (3) FID
    fid_parts = [(n, summary[n]["fid"]) for n in ("GAN-NN", "VAE-prior-NN")
                 if n in summary and "fid" in summary[n]]
    if fid_parts:
        s = "，".join(f"{n} = {v:.2f}" for n, v in fid_parts)
        lines.append(f"- **FID（越低越好）**：{s}。")
    else:
        lines.append("- **FID 未计算**（加 `--fid` 可计算，需联网下载 Inception 权重）。")

    lines += [
        "- **指标与感知的错位**：PSNR/SSIM 是逐像素指标，天然偏向「平滑但结构正确」的结果；"
        "GAN 的输出纹理锐利但像素位置不可能与 GT 完全对齐，因此在这两个指标上吃亏。"
        "这是「用 PSNR 评价生成模型不公平」的经典现象，报告里应同时给出定性对比图。",
        "",
        "## 3. 定性评估图",
        "",
        "- `compare_vae_recon.png`：上排 GT、下排 VAE 重建（一一对应，可直接比较）",
        "- `compare_gan_inversion.png`：上排 GT、下排 GAN 反演重建",
        "- `compare_gan_nn.png`：上排最近邻真实图、下排 GAN 生成图（无真实对应）",
        "- `samples_gan.png` / `samples_vae_prior.png`：两个模型的随机采样",
        "- `metric_bar.png`：PSNR/SSIM 柱状图",
        "",
        "> 定性观察要点：五官是否完整、有无棋盘格/网格伪影、肤色与背景是否自然、"
        "样本多样性（若 64 张高度相似，说明 GAN 出现模式崩塌或 VAE 后验塌缩）。",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
