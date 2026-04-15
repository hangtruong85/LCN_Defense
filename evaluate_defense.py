"""
Evaluations on Defense Capability
===================================
Evaluates LCN defense against:
  - IG  (Inverting Gradients, optimization-based)
  - iDLG (analytics-based)

Metrics: PSNR, SSIM, LPIPS
Conditions: No Defense / DP / LCN
"""

import os
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

from models import get_model
from attacks import ig_attack, idlg_attack, set_attack_logger
from metrics import compute_psnr, compute_ssim, compute_lpips
from lcn import apply_lcn, apply_dp_noise


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────
_log_file = None

def init_logger(log_path):
    """Open log file and register it globally."""
    global _log_file
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_file = open(log_path, "a", buffering=1)  # line-buffered
    _log(f"Log file: {log_path}")

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Tiny-ImageNet downloader
# ─────────────────────────────────────────────
def _download_tinyimagenet(data_root="./data"):
    """
    Download and extract Tiny-ImageNet-200 into data_root.
    Also fixes the val/ directory structure so ImageFolder can read it.
    Returns path to the val/ directory.
    """
    import urllib.request
    import zipfile
    import shutil

    url      = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    out_dir  = os.path.join(data_root, "tiny-imagenet-200")
    val_dir  = os.path.join(out_dir, "val")

    os.makedirs(data_root, exist_ok=True)

    # ── Progress callback (defined once, used everywhere) ─────
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb  = downloaded / 1024 / 1024
            if block_num % 500 == 0:
                _log(f"  {pct:.1f}%  ({mb:.1f} MB)")

    # ── Download ──────────────────────────────────────────────
    def _do_download():
        _log(f"Downloading Tiny-ImageNet from {url} ...")
        _log("(~240 MB, this may take a few minutes)")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        _log("Download complete.")

    if not os.path.exists(zip_path):
        _do_download()
    else:
        _log(f"Zip already exists: {zip_path}")

    # ── Validate zip, re-download if corrupt ──────────────────
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Bad file in zip: {bad}")
    except zipfile.BadZipFile:
        _log("Zip file is corrupt — deleting and re-downloading...")
        os.remove(zip_path)
        _do_download()

    if not os.path.exists(out_dir):
        _log(f"Extracting to {out_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        _log("Extraction complete.")
    else:
        _log(f"Already extracted: {out_dir}")

    # ── Fix val/ structure ────────────────────────────────────
    # Raw val/ has all images flat in val/images/ with a val_annotations.txt
    # ImageFolder needs: val/<class_name>/<image>.JPEG
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    img_dir  = os.path.join(val_dir, "images")

    if os.path.exists(ann_file) and os.path.exists(img_dir):
        _log("Reorganizing val/ directory for ImageFolder compatibility...")

        # Parse annotation file: filename \t class_id \t ...
        img_to_class = {}
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_to_class[parts[0]] = parts[1]

        # Move each image into val/<class>/
        moved = 0
        for img_name, cls in img_to_class.items():
            src = os.path.join(img_dir, img_name)
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            dst = os.path.join(cls_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1

        _log(f"Moved {moved} images into class subdirectories.")

        # Remove now-empty images/ dir and annotation file
        if os.path.exists(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)

    else:
        _log("val/ already reorganized — skipping.")

    return val_dir


# ─────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────
def get_dataset(name, n_samples=10, tinyimagenet_dir=None):
    """Return a small subset of the test set for reconstruction evaluation."""
    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform)
    elif name == "tinyimagenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                  (0.229, 0.224, 0.225)),
        ])
        # Auto-detect Tiny-ImageNet val directory
        candidates = (
            [tinyimagenet_dir] if tinyimagenet_dir else []
        ) + [
            "./data/tiny-imagenet-200/val",
            "./data/tiny-imagenet/val",
            "../data/tiny-imagenet-200/val",
            "/data/tiny-imagenet-200/val",
            "/dataset/tiny-imagenet-200/val",
            "/home/hangttt/data/tiny-imagenet-200/val",
        ]
        tiny_root = None
        for c in candidates:
            if os.path.isdir(c):
                tiny_root = c
                break
        if tiny_root is None:
            _log("Tiny-ImageNet not found locally — downloading...")
            tiny_root = _download_tinyimagenet("./data")

        _log(f"Tiny-ImageNet val dir: {tiny_root}")
        dataset = torchvision.datasets.ImageFolder(
            root=tiny_root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    indices = random.sample(range(len(dataset)), n_samples)
    return Subset(dataset, indices)


# ─────────────────────────────────────────────
# BatchNorm fix: set BN layers to eval while keeping others trainable
# ─────────────────────────────────────────────
def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────
# Simulate one FL round and return (model, true_grad)
# ─────────────────────────────────────────────
def simulate_fl_round(model, image, label, device, prev_grad=None):
    model.train()
    _set_bn_eval(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    out  = model(image.unsqueeze(0).to(device))
    loss = criterion(out, label.unsqueeze(0).to(device))
    loss.backward()

    true_grad = [p.grad.clone().detach() for p in model.parameters()]
    optimizer.step()
    return model, true_grad


# ─────────────────────────────────────────────
# Evaluate one (image, label) under one defense condition
# ─────────────────────────────────────────────
def evaluate_single(model, image, label, device, defense, param,
                    attack_type, prev_grad, lpips_fn, img_idx,
                    n_images, cond_label, save_dir=None,
                    model_name="mobilenet", img_size=32):
    model_copy = get_model(model_name,
                           num_classes=_num_classes(model),
                           img_size=img_size).to(device)
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()

    # True gradient
    _log(f"      [{cond_label}] img {img_idx}/{n_images} — computing true gradient...")
    _, true_grad = simulate_fl_round(
        get_model(model_name,
                  num_classes=_num_classes(model),
                  img_size=img_size).to(device),
        image, label, device)

    # Apply defense
    _log(f"      [{cond_label}] img {img_idx}/{n_images} — applying defense: {defense} ...")
    if defense == "none":
        observed_grad = true_grad
    elif defense == "dp":
        observed_grad = apply_dp_noise(true_grad, sigma=param, device=device)
    elif defense == "lcn":
        assert prev_grad is not None, "LCN requires prev_grad"
        observed_grad = apply_lcn(true_grad, prev_grad, alpha=param)
    else:
        raise ValueError(f"Unknown defense: {defense}")

    # Attack
    _log(f"      [{cond_label}] img {img_idx}/{n_images} — running {attack_type.upper()} attack...")
    t0 = time.time()
    if attack_type == "ig":
        reconstructed = ig_attack(model_copy, observed_grad, image, device)
    elif attack_type == "idlg":
        reconstructed = idlg_attack(model_copy, observed_grad, image,
                                     label, device)
    else:
        raise ValueError(f"Unknown attack: {attack_type}")
    atk_elapsed = time.time() - t0
    _log(f"      [{cond_label}] img {img_idx}/{n_images} — attack done in {atk_elapsed:.1f}s")

    # Metrics
    orig_np = _to_numpy(image)
    rec_np  = _to_numpy(reconstructed)
    psnr    = compute_psnr(orig_np, rec_np)
    ssim    = compute_ssim(orig_np, rec_np)
    lpips   = compute_lpips(image, reconstructed, lpips_fn, device)

    _log(f"      [{cond_label}] img {img_idx}/{n_images} — "
         f"PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}")

    # Save images
    if save_dir is not None:
        _save_image_pair(image, reconstructed, save_dir,
                         defense, param, img_idx, attack_type)
        _log(f"      [{cond_label}] img {img_idx}/{n_images} — images saved.")

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _to_numpy(tensor):
    t = tensor.detach().cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t.permute(1, 2, 0).numpy()


def _num_classes(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    return 10


def _save_image_pair(orig_tensor, rec_tensor, save_dir,
                     defense, param, img_idx, attack_type):
    """
    Save original and reconstructed images at native resolution (no resize).
    Folder structure:
        save_dir/
            orig/        img_001.png  ...
            <cond>/      img_001.png  ...
            comparison/  img_001_<cond>.png  (side-by-side, no padding)
    """
    from PIL import Image as PILImage
    import numpy as np

    def _to_pil(t):
        """CHW tensor → PIL Image at native resolution."""
        t = t.detach().cpu().float()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        t = t.clamp(0, 1)
        # CHW → HWC, scale to uint8
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return PILImage.fromarray(arr)

    param_str = str(param).replace(".", "p") if param is not None else "none"
    cond_str  = f"{defense}_{param_str}"

    orig_dir  = os.path.join(save_dir, "orig")
    rec_dir   = os.path.join(save_dir, cond_str)
    cmp_dir   = os.path.join(save_dir, "comparison")
    for d in [orig_dir, rec_dir, cmp_dir]:
        os.makedirs(d, exist_ok=True)

    fname     = f"img_{img_idx:03d}_{attack_type}.png"
    cmp_fname = f"img_{img_idx:03d}_{cond_str}_{attack_type}.png"

    orig_pil = _to_pil(orig_tensor)
    rec_pil  = _to_pil(rec_tensor)

    # Save individual images at exact native resolution
    orig_pil.save(os.path.join(orig_dir, fname))
    rec_pil.save(os.path.join(rec_dir,  fname))

    # Side-by-side comparison: concat horizontally, NO padding, NO resize
    W, H = orig_pil.size
    cmp  = PILImage.new("RGB", (W * 2, H))
    cmp.paste(orig_pil, (0,  0))
    cmp.paste(rec_pil,  (W,  0))
    cmp.save(os.path.join(cmp_dir, cmp_fname))


# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────
def run_evaluation(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        ts      = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"defense_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Init log file inside run_dir
    log_filename = (f"defense_{args.dataset}_{args.model}_{args.attack}.log")
    log_path = os.path.join(run_dir, log_filename)
    init_logger(log_path)
    set_attack_logger(_log_file)  # share handle with attacks.py

    _log(f"Run directory: {run_dir}")
    _log(f"Device: {device}")
    _log(f"Args: {vars(args)}")

    _log("Loading LPIPS model (alex)...")
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    _log("LPIPS ready.")

    _log(f"Loading dataset: {args.dataset}  (n_samples={args.n_samples})")
    tiny_dir = getattr(args, "tinyimagenet_dir", None)
    dataset = get_dataset(args.dataset, n_samples=args.n_samples,
                          tinyimagenet_dir=tiny_dir)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)
    _log(f"Dataset ready: {len(dataset)} images selected.")

    num_classes = 10 if args.dataset == "cifar10" else 200
    _log(f"Building model: {args.model}  num_classes={num_classes}")
    img_size = 32 if args.dataset == "cifar10" else 64
    model = get_model(args.model, num_classes=num_classes,
                      img_size=img_size).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        _log(f"Loaded checkpoint: {args.checkpoint}")
    else:
        _log("No checkpoint found — using random weights (demo mode).")

    conditions = [
        ("none",  None),
        ("dp",    1e-3),
        ("dp",    1e-2),
        ("lcn",   1.1),
        ("lcn",   0.9),
        ("lcn",   0.7),
        ("lcn",   0.5),
    ]

    n_conditions = len(conditions)
    n_images     = args.n_samples
    results      = []
    total_start  = time.time()

    _log(f"\nTotal plan: 1 attack × {n_conditions} conditions × {n_images} images "
         f"= {n_conditions * n_images} jobs")

    for atk in [args.attack]:
        _log(f"\n{'='*64}")
        _log(f"ATTACK: {atk.upper()}")
        _log(f"{'='*64}")

        for cond_idx, (defense, param) in enumerate(conditions):
            label_str = (f"sigma={param}" if defense == "dp"  else
                         f"alpha={param}" if defense == "lcn" else "no-defense")
            cond_label = f"{defense}/{label_str}"

            _log(f"\n  ── Condition {cond_idx+1}/{n_conditions}: "
                 f"{cond_label} ──")
            cond_start = time.time()

            psnr_list, ssim_list, lpips_list = [], [], []
            prev_grad_store = {}

            for idx, (image, label) in enumerate(loader):
                image = image.squeeze(0)
                label = label.squeeze(0)

                if defense == "lcn":
                    if idx not in prev_grad_store:
                        _log(f"      [lcn] img {idx+1} — simulating prev round (t-1)...")
                        _, pg = simulate_fl_round(
                            get_model(args.model,
                                      num_classes=num_classes,
                                      img_size=img_size).to(device),
                            image, label, device)
                        prev_grad_store[idx] = pg
                    prev_grad = prev_grad_store[idx]
                else:
                    prev_grad = None

                img_save_dir = os.path.join(
                    run_dir, "images",
                    f"{args.dataset}_{args.model}_{atk}")
                metrics = evaluate_single(
                    model, image, label, device,
                    defense, param, atk, prev_grad, lpips_fn,
                    img_idx=idx+1, n_images=n_images,
                    cond_label=cond_label, save_dir=img_save_dir,
                    model_name=args.model, img_size=img_size)

                psnr_list.append(metrics["psnr"])
                ssim_list.append(metrics["ssim"])
                lpips_list.append(metrics["lpips"])

            cond_elapsed = time.time() - cond_start
            _log(f"  Condition {cond_idx+1}/{n_conditions} COMPLETE "
                 f"({cond_elapsed:.1f}s)  |  "
                 f"mean PSNR={np.mean(psnr_list):.2f}  "
                 f"mean SSIM={np.mean(ssim_list):.4f}  "
                 f"mean LPIPS={np.mean(lpips_list):.4f}")

            results.append({
                "attack":     atk,
                "defense":    defense,
                "param":      str(param),
                "dataset":    args.dataset,
                "model":      args.model,
                "PSNR_mean":  np.mean(psnr_list),
                "PSNR_std":   np.std(psnr_list),
                "SSIM_mean":  np.mean(ssim_list),
                "SSIM_std":   np.std(ssim_list),
                "LPIPS_mean": np.mean(lpips_list),
                "LPIPS_std":  np.std(lpips_list),
            })

    # Save CSV
    csv_path = os.path.join(run_dir,
                            f"defense_capability_{args.dataset}_{args.model}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - total_start
    _log(f"\nAll done in {total_elapsed/60:.1f} min. Results saved to: {csv_path}")

    if _log_file is not None:
        _log_file.close()

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Defense Capability Evaluation for LCN")
    parser.add_argument("--dataset",    default="cifar10",
                        choices=["cifar10", "tinyimagenet"])
    parser.add_argument("--model",      default="mobilenet",
                        choices=["mobilenet", "resnet18", "lenet"])
    parser.add_argument("--attack",     default="ig",
                        choices=["ig", "idlg"])
    parser.add_argument("--n_samples",  type=int, default=10)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pretrained model .pth file")
    parser.add_argument("--tinyimagenet_dir", default=None,
                        help="Path to Tiny-ImageNet val directory "
                             "(e.g. /data/tiny-imagenet-200/val)")
    parser.add_argument("--output_dir", default="./results",
                        help="Base results dir (ignored if --run_dir is set)")
    parser.add_argument("--run_dir",    default=None,
                        help="Explicit output directory for this run "
                             "(e.g. results/defense_20260412_221355). "
                             "If not set, auto-created under output_dir.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_evaluation(args)