"""
visualize.py — Generate Fig. defense_ig_visual
Shows: original | no-defense | DP | LCN reconstructions side-by-side
"""

import os
import argparse
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

from models import get_model
from attacks import ig_attack, idlg_attack
from lcn import apply_lcn, apply_dp_noise, generate_betas
from evaluate_defense import simulate_fl_round, _to_numpy, set_seed


# ─────────────────────────────────────────────
# Unnormalize helper
# ─────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
TINY_MEAN    = (0.485, 0.456, 0.406)
TINY_STD     = (0.229, 0.224, 0.225)

def unnormalize(tensor, mean, std):
    t = tensor.clone().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


# ─────────────────────────────────────────────
# Build visualization figure
# ─────────────────────────────────────────────
def build_figure(args):
    set_seed(args.seed)   # seed cho gradient/model reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed riêng cho việc chọn ảnh — None = random mỗi lần
    import time as _time
    vis_seed = args.vis_seed if args.vis_seed is not None                else int(_time.time()) % 100000
    random.seed(vis_seed)
    print(f"[visualize] vis_seed={vis_seed}  "
          f"(dùng --vis_seed {vis_seed} để reproduce)")

    # Dataset
    if args.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform)
    else:
        mean, std = TINY_MEAN, TINY_STD
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.ImageFolder(
            root="./data/tiny-imagenet-200/val", transform=transform)

    # Pick n_show images
    indices = random.sample(range(len(dataset)), args.n_show)

    # Model
    num_classes = 10 if args.dataset == "cifar10" else 200
    img_size = 32 if args.dataset == "cifar10" else 64
    model = get_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device))
    model.eval()

    # LPIPS (not used in viz, but needed for evaluate_single import)
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    except ImportError:
        lpips_fn = None

    # Conditions: (label, defense, param)
    conditions = [
        ("No Defense",    "none",  None),
        ("DP  σ=1e-3",    "dp",    1e-3),
        ("DP  σ=1e-2",    "dp",    1e-2),
        ("LCN α=0.9",     "lcn",   0.9),
        ("LCN α=0.7",     "lcn",   0.7),
        ("LCN α=0.5",     "lcn",   0.5),
    ]
    n_cols = 1 + len(conditions)   # original + each condition
    n_rows = args.n_show

    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols,
                            hspace=0.05, wspace=0.05)

    col_labels = ["Original"] + [c[0] for c in conditions]

    for row, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image  # CHW tensor
        label = torch.tensor(label)

        # Obtain prev_grad for LCN by simulating round t-1
        m_prev = get_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
        m_prev.load_state_dict(model.state_dict())
        _, prev_grad, w_prev, _ = simulate_fl_round(m_prev, image, label, device)

        # True gradient at round t
        m_cur = get_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
        m_cur.load_state_dict(model.state_dict())
        _, true_grad, w_cur, _ = simulate_fl_round(m_cur, image, label, device)

        # Plot original
        ax = fig.add_subplot(gs[row, 0])
        img_show = unnormalize(image, mean, std).permute(1, 2, 0).numpy()
        ax.imshow(img_show)
        ax.axis("off")
        if row == 0:
            ax.set_title("Original", fontsize=8, fontweight="bold")

        # Sinh beta cho LCN (trusted third party, 1 lần per ảnh)
        betas  = generate_betas(num_classes if args.model == "lenet" else 5, seed=42)
        beta_k = betas[0]
        m_lcn  = 5  # số clients

        # Plot each defense condition
        for col, (cond_label, defense, param) in enumerate(conditions):
            if defense == "none":
                obs_grad = true_grad
            elif defense == "dp":
                obs_grad = apply_dp_noise(true_grad, sigma=param,
                                          device=device)
            elif defense == "lcn":
                obs_grad = apply_lcn(
                    true_grad=true_grad,
                    w_cur=w_cur,
                    w_prev=w_prev,
                    alpha=param,
                    beta_k=beta_k,
                    m=m_lcn
                )

            m_atk = get_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
            m_atk.load_state_dict(model.state_dict())
            m_atk.eval()

            if args.attack == "ig":
                rec = ig_attack(m_atk, obs_grad, image, device, original_label=label,
                                n_iter=args.ig_iter)
            else:
                rec = idlg_attack(m_atk, obs_grad, image, label, device)

            rec_show = unnormalize(rec, mean, std).permute(1, 2, 0).numpy()
            rec_show = rec_show.clip(0, 1)

            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(rec_show)
            ax.axis("off")
            if row == 0:
                ax.set_title(cond_label, fontsize=7, fontweight="bold")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"fig_defense_{args.attack}_{args.dataset}_{args.model}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

    # Save PNG at same size as PDF (dpi=72 = screen resolution, no upscaling)
    fig.savefig(out_path.replace(".pdf", ".png"),
                bbox_inches="tight", dpi=72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualization of defense capability")
    parser.add_argument("--dataset",    default="cifar10",
                        choices=["cifar10", "tinyimagenet"])
    parser.add_argument("--model",      default="mobilenet",
                        choices=["mobilenet", "resnet18", "lenet"])
    parser.add_argument("--attack",     default="ig",
                        choices=["ig", "idlg"])
    parser.add_argument("--n_show",     type=int, default=4,
                        help="Number of images to show per row")
    parser.add_argument("--ig_iter",    type=int, default=2000,
                        help="IG iterations (lower for quick preview)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="./results/figures")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Seed cho evaluate (reproduce gradient)")
    parser.add_argument("--vis_seed", type=int, default=None,
                        help="Seed cho chọn ảnh visualize "
                             "(None = random mỗi lần, khác nhau giữa các run)")
    args = parser.parse_args()

    build_figure(args)