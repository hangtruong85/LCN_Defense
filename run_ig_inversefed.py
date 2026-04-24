"""
run_ig_inversefed.py
====================
Đánh giá defense capability với IG attack dùng code GỐC của nhóm tác giả
(Geiping et al. 2020 — inversefed.GradientReconstructor).

Bổ sung DP và LCN defense để so sánh, lưu CSV + ảnh tổng hợp.

API inversefed (từ code gốc):
    rec_machine = GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=...)
    # input_gradient: list of tensors (từng param gradient)
    # labels        : torch.Tensor shape (num_images,) — ground-truth label
    # output        : tensor (1, C, H, W) — normalized

boxed clamp trong code gốc:
    x_trial.data = torch.max(torch.min(x_trial, (1-dm)/ds), -dm/ds)
    → đây là cách đúng, dùng mean/std thực của dataset

Dùng:
    python run_ig_inversefed.py --model ResNet18 --dataset cifar10 \\
        --checkpoint ./checkpoints/cifar10_resnet18.pth --n_samples 10

    python run_ig_inversefed.py --model MobileNet --dataset cifar10 \\
        --checkpoint ./checkpoints/cifar10_mobilenet.pth --n_samples 10 --quick
"""

import os, sys, csv, time, random, argparse, datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image as PILImage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── inversefed ────────────────────────────────────────────────
try:
    import inversefed
    from inversefed.reconstruction_algorithms import GradientReconstructor
    from inversefed import consts
except ImportError:
    print("ERROR: inversefed chưa cài.\n"
          "  cd <inversefed_repo> && pip install -e .")
    sys.exit(1)

# ── LCN / DP từ code hiện có ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcn import apply_lcn, apply_dp_noise, generate_betas

# ─────────────────────────────────────────────────────────────
# Dataset configs — dùng đúng const của inversefed
# ─────────────────────────────────────────────────────────────
DATASET_META = {
    "cifar10": {
        "num_classes": 10,
        "img_size":    32,
        "channel":     3,
        "mean": consts.cifar10_mean,
        "std":  consts.cifar10_std,
    },
    "tinyimagenet": {
        "num_classes": 200,
        "img_size":    64,
        "channel":     3,
        "mean": consts.imagenet_mean,
        "std":  consts.imagenet_std,
    },
}

# Defense conditions
CONDITIONS = [
    ("No Defense", "none",  None),
    ("DP σ=1e-3",  "dp",    0.001),
    ("DP σ=1e-2",  "dp",    0.01),
    ("LCN α=0.9",  "lcn",   0.9),
    ("LCN α=0.7",  "lcn",   0.7),
    ("LCN α=0.5",  "lcn",   0.5),
]

COND_DIR = {
    "No Defense": "none_None",
    "DP σ=1e-3":  "dp_0p001",
    "DP σ=1e-2":  "dp_0p01",
    "LCN α=0.9":  "lcn_0p9",
    "LCN α=0.7":  "lcn_0p7",
    "LCN α=0.5":  "lcn_0p5",
}

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
_log_fh = None

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
def get_dataset(name, n_samples, tinyimagenet_dir=None):
    meta = DATASET_META[name]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])
    if name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=tf)
    elif name == "tinyimagenet":
        d = tinyimagenet_dir or "./data/tiny-imagenet-200/val"
        if not os.path.isdir(d):
            raise FileNotFoundError(f"TinyImageNet not found: {d}")
        ds = torchvision.datasets.ImageFolder(d, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    actual  = min(n_samples, len(ds))
    indices = random.sample(range(len(ds)), actual)
    _log(f"Dataset {name}: {len(ds)} total → chọn {actual} ảnh")
    return Subset(ds, indices)

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
def get_model(name, num_classes, checkpoint=None, device="cpu"):
    name_l = name.lower().replace("-", "").replace("_", "")
    if name_l == "resnet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, num_classes)
    elif name_l in ("mobilenet", "mobilenetv2"):
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
    elif "resnet20" in name_l or "convnet" in name_l:
        try:
            model, _ = inversefed.construct_model(
                name, num_classes=num_classes, num_channels=3)
        except Exception as e:
            _log(f"WARNING: construct_model({name}) failed: {e}")
            _log("Fallback → ResNet18")
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Supported: ResNet18, MobileNet, ResNet20-4, ConvNet")

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        # Xử lý state dict có prefix 'module.' (DataParallel)
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        _log(f"Loaded checkpoint: {checkpoint}")
    else:
        _log(f"No checkpoint → random weights")

    model.to(device)
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────
# Unnormalize tensor → HWC numpy [0,1]
# ─────────────────────────────────────────────────────────────
def unnorm(tensor, mean, std):
    t = tensor.detach().cpu().float()
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def _psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)))

def _ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        return 0.0

def _lpips(a_t, b_t, lpips_fn, device):
    if lpips_fn is None:
        return 0.0
    try:
        import torch.nn.functional as F
        def prep(t):
            t = t.float().unsqueeze(0).to(device)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8) * 2 - 1
            if t.shape[-1] < 32:
                t = F.interpolate(t, (32,32), mode="bilinear",
                                  align_corners=False)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            return t
        with torch.no_grad():
            return float(lpips_fn(prep(a_t), prep(b_t)).item())
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────
# Apply defense
# ─────────────────────────────────────────────────────────────
def apply_defense(true_grad, defense, param, lcn_state, device):
    """
    true_grad : list of tensors (output của torch.autograd.grad)
    Trả về list of tensors đã được perturb.
    """
    if defense == "none":
        return list(true_grad)
    elif defense == "dp":
        return apply_dp_noise(list(true_grad), sigma=param, device=device)
    elif defense == "lcn":
        return apply_lcn(
            true_grad=list(true_grad),
            w_cur    =lcn_state["w_cur"],
            w_prev   =lcn_state["w_prev"],
            alpha    =param,
            beta_k   =lcn_state["beta_k"],
            m        =lcn_state["m"],
        )
    else:
        raise ValueError(f"Unknown defense: {defense}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run(args):
    global _log_fh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta   = DATASET_META[args.dataset]

    # dm, ds: tensors shape (C,1,1) — đúng như code gốc tác giả
    dm = torch.as_tensor(meta["mean"], device=device, dtype=torch.float)[:, None, None]
    ds = torch.as_tensor(meta["std"],  device=device, dtype=torch.float)[:, None, None]

    # Run dir
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = f"{args.dataset}_{args.model.lower().replace('-','')}"
    run_dir = os.path.join(args.output_dir, f"ig_inv_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    _log_fh = open(os.path.join(run_dir, f"log_{tag}.txt"), "w")

    _log(f"Run dir : {run_dir}")
    _log(f"Dataset : {args.dataset}  Model: {args.model}  Device: {device}")
    _log(f"Restarts: {args.restarts}  iter: {args.ig_iter}  cost_fn: {args.cost_fn}")

    # Dataset & model
    random.seed(None)
    dataset = get_dataset(args.dataset, args.n_samples,
                          tinyimagenet_dir=args.tinyimagenet_dir)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    model = get_model(args.model, meta["num_classes"],
                      checkpoint=args.checkpoint, device=device)

    # inversefed config — đúng theo paper gốc
    inv_config = dict(
        signed         = True,
        boxed          = True,
        cost_fn        = args.cost_fn,   # "sim" = cosine (paper gốc)
        indices        = "def",
        weights        = "equal",
        lr             = 0.1,
        optim          = "adam",
        restarts       = args.restarts,
        max_iterations = args.ig_iter,
        total_variation= args.tv,
        init           = "randn",
        filter         = "none",
        lr_decay       = True,
        scoring_choice = "loss",
    )
    _log(f"IG config: {inv_config}")

    # LPIPS
    try:
        import lpips as _lpips_lib
        lpips_fn = _lpips_lib.LPIPS(net="alex").to(device)
        _log("LPIPS: AlexNet loaded")
    except Exception:
        lpips_fn = None
        _log("WARNING: lpips not available → LPIPS=0")

    # LCN betas
    n_clients = 5
    betas     = generate_betas(n_clients, seed=args.seed)
    beta_k    = betas[0]
    _log(f"LCN betas: {[f'{b:.4f}' for b in betas]}  beta_k={beta_k:.4f}")

    # Image dirs
    img_root = os.path.join(run_dir, "images", f"{tag}_ig_inv")
    os.makedirs(os.path.join(img_root, "orig"), exist_ok=True)
    for cname in COND_DIR.values():
        os.makedirs(os.path.join(img_root, cname), exist_ok=True)

    # Metrics accumulator
    acc = {c[0]: {"psnr": [], "ssim": [], "lpips": []} for c in CONDITIONS}

    # w_prev cho LCN
    w_global_prev = [p.detach().clone() for p in model.parameters()]

    # ── Main loop ──────────────────────────────────────────
    for img_idx, (image, label) in enumerate(loader):
        image = image.squeeze(0).to(device)   # (C,H,W)
        label = label.to(device)              # scalar tensor

        _log(f"\n--- Image {img_idx+1}/{len(dataset)}  "
             f"label={label.item()} ---")

        # w_cur
        w_global_cur = [p.detach().clone() for p in model.parameters()]
        lcn_state = {
            "w_cur":  w_global_cur,
            "w_prev": w_global_prev,
            "beta_k": beta_k,
            "m":      n_clients,
        }

        # Tính true gradient — đúng cách tác giả dùng
        model.zero_grad()
        img_in = image.unsqueeze(0)                  # (1,C,H,W)
        lbl_in = label.reshape(1).long()             # (1,)

        loss = torch.nn.CrossEntropyLoss()(
            model(img_in), lbl_in)
        true_grad = torch.autograd.grad(
            loss, model.parameters(), create_graph=False)
        true_grad = [g.detach() for g in true_grad]

        full_norm = torch.stack([g.norm() for g in true_grad]).mean()
        _log(f"  Full gradient norm: {full_norm:.4e}")

        # Lưu ảnh gốc
        orig_np  = unnorm(image, meta["mean"], meta["std"])
        orig_pil = PILImage.fromarray(
            (orig_np * 255).astype(np.uint8))
        orig_pil.save(os.path.join(
            img_root, "orig", f"img_{img_idx+1:03d}_ig.png"))

        # ── Vòng lặp conditions ───────────────────────────
        for cname, defense, param in CONDITIONS:
            _log(f"  [{cname}] applying defense ...")
            obs_grad = apply_defense(true_grad, defense, param,
                                     lcn_state, device)

            _log(f"  [{cname}] GradientReconstructor "
                 f"(restarts={args.restarts}, iter={args.ig_iter}) ...")
            t0 = time.time()

            # Tạo mới rec_machine mỗi lần để tránh state cũ
            rec_machine = GradientReconstructor(
                model,
                (dm, ds),
                inv_config,
                num_images=1,
            )

            img_shape = (meta["channel"],
                         meta["img_size"],
                         meta["img_size"])

            # reconstruct: labels = ground-truth (1,) tensor
            output, stats = rec_machine.reconstruct(
                obs_grad,
                lbl_in,
                img_shape=img_shape,
                dryrun=False,
            )
            # output: (1,C,H,W) normalized, trên device của model

            elapsed = time.time() - t0
            _log(f"  [{cname}] done {elapsed:.1f}s  "
                 f"rec_loss={stats['opt']:.4f}")

            # Unnormalize → metrics
            rec_t  = output.squeeze(0).cpu()
            rec_np = unnorm(rec_t, meta["mean"], meta["std"])

            p  = _psnr(orig_np, rec_np)
            s  = _ssim(orig_np, rec_np)
            lp = _lpips(image.cpu(), rec_t, lpips_fn, device)

            acc[cname]["psnr"].append(p)
            acc[cname]["ssim"].append(s)
            acc[cname]["lpips"].append(lp)
            _log(f"  [{cname}] PSNR={p:.2f}  SSIM={s:.4f}  LPIPS={lp:.4f}")

            # Lưu ảnh
            rec_pil = PILImage.fromarray(
                (rec_np * 255).astype(np.uint8))
            rec_pil.save(os.path.join(
                img_root, COND_DIR[cname],
                f"img_{img_idx+1:03d}_ig.png"))

        w_global_prev = w_global_cur

    # ── CSV ────────────────────────────────────────────────
    csv_path = os.path.join(run_dir,
                    f"defense_capability_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "defense", "param",
                    "PSNR_mean", "PSNR_std",
                    "SSIM_mean", "SSIM_std",
                    "LPIPS_mean","LPIPS_std"])
        for cname, defense, param in CONDITIONS:
            m = acc[cname]
            w.writerow([
                cname, defense, str(param),
                f"{np.mean(m['psnr']):.4f}", f"{np.std(m['psnr']):.4f}",
                f"{np.mean(m['ssim']):.4f}", f"{np.std(m['ssim']):.4f}",
                f"{np.mean(m['lpips']):.4f}",f"{np.std(m['lpips']):.4f}",
            ])
    _log(f"CSV saved: {csv_path}")

    # ── Figure ─────────────────────────────────────────────
    _make_figure(img_root, run_dir, tag, n_show=min(4, len(dataset)))
    _log(f"\nAll done. Run dir: {run_dir}")
    if _log_fh:
        _log_fh.close()


def _make_figure(img_root, run_dir, tag, n_show=4):
    import glob
    orig_files = sorted(
        glob.glob(os.path.join(img_root, "orig", "*.png")))[:n_show]
    if not orig_files:
        return

    cols   = [c[0] for c in CONDITIONS]
    cdirs  = [COND_DIR[c] for c in cols]
    n_rows = len(orig_files)
    n_cols = 1 + len(cdirs)

    fig = plt.figure(figsize=(2.2*n_cols, 2.2*n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols,
                             hspace=0.04, wspace=0.04)

    for row, op in enumerate(orig_files):
        fname = os.path.basename(op)
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(PILImage.open(op).convert("RGB"))
        ax.axis("off")
        if row == 0:
            ax.set_title("Original", fontsize=7,
                         fontweight="bold", pad=3)

        for col, (cd, cl) in enumerate(zip(cdirs, cols)):
            ax = fig.add_subplot(gs[row, col+1])
            rp = os.path.join(img_root, cd, fname)
            if os.path.exists(rp):
                ax.imshow(PILImage.open(rp).convert("RGB"))
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes,
                        color="#888")
            ax.axis("off")
            if row == 0:
                ax.set_title(cl, fontsize=6,
                             fontweight="bold", pad=3)

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, f"fig_ig_inv_{tag}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    _log(f"Figure: {base}.pdf + .png")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="IG defense evaluation — inversefed (code gốc tác giả)")
    p.add_argument("--model",      default="ResNet18",
        help="ResNet18 | MobileNet | ResNet20-4 | ConvNet")
    p.add_argument("--dataset",    default="cifar10",
        choices=list(DATASET_META.keys()))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n_samples",  type=int, default=10)
    p.add_argument("--restarts",   type=int, default=8,
        help="IG restarts (paper gốc: 8)")
    p.add_argument("--ig_iter",    type=int, default=24_000,
        help="IG max_iterations (paper gốc: 24000)")
    p.add_argument("--cost_fn",    default="sim",
        choices=["sim","l2","l1","max","simlocal"],
        help="cost function (paper gốc: sim = cosine)")
    p.add_argument("--tv",         type=float, default=1e-4)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--tinyimagenet_dir", default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--quick",      action="store_true",
        help="Quick test: n_samples=2, iter=500, restarts=1")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.n_samples = 2
        args.ig_iter   = 500
        args.restarts  = 1
        print(">>> QUICK MODE: n_samples=2, iter=500, restarts=1")
    torch.manual_seed(args.seed)
    run(args)