"""
train_fl.py — Federated Learning Training with LCN Defense
============================================================
Trains a global model using FedAvg (and optionally LCN/DP defense)
then saves the checkpoint for use in defense capability evaluation.

Usage examples:
  # Train baseline FedAvg (no defense) — recommended for evaluation
  python train_fl.py --dataset cifar10   --model mobilenet  --defense none
  python train_fl.py --dataset tinyimagenet --model resnet18 --defense none

  # Train with LCN defense
  python train_fl.py --dataset cifar10 --model mobilenet --defense lcn --alpha 0.7

  # Train with DP defense
  python train_fl.py --dataset cifar10 --model mobilenet --defense dp --sigma 1e-3

Checkpoints saved to:
  ./checkpoints/<dataset>_<model>_<defense>.pth   (best val accuracy)
  ./checkpoints/<dataset>_<model>_<defense>_last.pth  (last round)
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
from torch.utils.data import DataLoader, Subset

from models import get_model
from lcn import apply_lcn, apply_dp_noise


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
_log_file = None

def init_logger(log_path):
    global _log_file
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_file = open(log_path, "a", buffering=1)
    _log(f"Log file: {log_path}")

def _log(msg):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
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
# BatchNorm fix for batch_size=1
# ─────────────────────────────────────────────
def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def get_datasets(name, n_clients, data_root="./data"):
    """
    Return (list_of_train_subsets, test_dataset).
    Training data is split IID across n_clients.
    """
    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = torchvision.datasets.CIFAR10(
            root=data_root, train=True,  download=True, transform=train_tf)
        test_ds    = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_tf)
        num_classes = 10

    elif name == "tinyimagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # Auto-detect or download Tiny-ImageNet
        train_dir, val_dir = _get_tinyimagenet_dirs(data_root)
        train_full = torchvision.datasets.ImageFolder(
            root=train_dir, transform=train_tf)
        test_ds    = torchvision.datasets.ImageFolder(
            root=val_dir, transform=test_tf)
        num_classes = 200

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # IID split across clients
    n        = len(train_full)
    per      = n // n_clients
    indices  = list(range(n))
    random.shuffle(indices)
    client_datasets = [
        Subset(train_full, indices[i * per: (i + 1) * per])
        for i in range(n_clients)
    ]

    return client_datasets, test_ds, num_classes


# ─────────────────────────────────────────────
# Tiny-ImageNet helpers
# ─────────────────────────────────────────────
def _get_tinyimagenet_dirs(data_root):
    import shutil, zipfile, urllib.request

    out_dir   = os.path.join(data_root, "tiny-imagenet-200")
    train_dir = os.path.join(out_dir, "train")
    val_dir   = os.path.join(out_dir, "val")
    zip_path  = os.path.join(data_root, "tiny-imagenet-200.zip")
    url       = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    os.makedirs(data_root, exist_ok=True)

    def _progress(block_num, block_size, total_size):
        if total_size > 0 and block_num % 500 == 0:
            mb  = block_num * block_size / 1024 / 1024
            pct = block_num * block_size / total_size * 100
            _log(f"  Downloading: {pct:.1f}%  ({mb:.1f} MB)")

    def _do_download():
        _log(f"Downloading Tiny-ImageNet (~240 MB)...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        _log("Download complete.")

    # Download if needed
    if not os.path.exists(zip_path):
        _do_download()
    else:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile
        except zipfile.BadZipFile:
            _log("Corrupt zip — re-downloading...")
            os.remove(zip_path)
            _do_download()

    # Extract if needed
    if not os.path.exists(out_dir):
        _log("Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        _log("Extraction complete.")

    # Fix val/ structure
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    img_dir  = os.path.join(val_dir, "images")
    if os.path.exists(ann_file) and os.path.exists(img_dir):
        _log("Reorganizing val/ for ImageFolder...")
        img_to_cls = {}
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_to_cls[parts[0]] = parts[1]
        moved = 0
        for img_name, cls in img_to_cls.items():
            src = os.path.join(img_dir, img_name)
            dst_dir = os.path.join(val_dir, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
        if moved:
            _log(f"Moved {moved} val images.")
        if os.path.exists(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)

    return train_dir, val_dir


# ─────────────────────────────────────────────
# One local training step (1 client, E epochs)
# ─────────────────────────────────────────────
def local_train(model, dataset, device, local_bs, local_epochs, lr):
    """
    Train model locally for local_epochs epochs.
    Returns local update Delta_w = w_local - w_global.
    """
    model.train()
    _set_bn_eval(model)

    loader    = DataLoader(dataset, batch_size=local_bs,
                           shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    w_init = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            _set_bn_eval(model)

    # Compute delta = w_local - w_global
    delta = {k: model.state_dict()[k].float() - w_init[k].float()
             for k in w_init}
    return delta


# ─────────────────────────────────────────────
# Apply defense to a list of deltas
# ─────────────────────────────────────────────
def apply_defense_to_deltas(deltas, prev_deltas, defense, alpha, sigma, device):
    """
    deltas      : list of dicts {param_name: tensor}
    prev_deltas : same structure, from previous round (for LCN)
    Returns defended deltas (same structure).
    """
    if defense == "none":
        return deltas

    defended = []
    for i, delta in enumerate(deltas):
        # Flatten dict → list of tensors for lcn/dp helpers
        keys   = list(delta.keys())
        values = [delta[k] for k in keys]

        if defense == "dp":
            noisy = apply_dp_noise(values, sigma=sigma, device=device)
            defended.append(dict(zip(keys, noisy)))

        elif defense == "lcn":
            if prev_deltas is not None:
                prev_values = [prev_deltas[i][k] for k in keys]
                mixed = apply_lcn(values, prev_values, alpha=alpha)
            else:
                # Round 0: no prev available, fall back to no defense
                mixed = values
            defended.append(dict(zip(keys, mixed)))

    return defended


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_ds, device, batch_size=128):
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger
    os.makedirs(args.output_dir, exist_ok=True)
    log_name = (f"train_{args.dataset}_{args.model}_"
                f"{args.defense}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    init_logger(os.path.join(args.output_dir, log_name))

    _log(f"Device  : {device}")
    _log(f"Args    : {vars(args)}")

    # Data
    _log(f"Loading dataset: {args.dataset}  n_clients={args.n_clients}")
    client_datasets, test_ds, num_classes = get_datasets(
        args.dataset, args.n_clients, data_root="./data")
    _log(f"num_classes={num_classes}  "
         f"train per client≈{len(client_datasets[0])}  "
         f"test={len(test_ds)}")

    # Model
    _log(f"Building model: {args.model}  num_classes={num_classes}")
    global_model = get_model(args.model, num_classes=num_classes).to(device)

    # Checkpoint paths
    ckpt_dir  = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    tag       = f"{args.dataset}_{args.model}"   # used by evaluate_defense.py
    best_path = os.path.join(ckpt_dir, f"{tag}.pth")
    last_path = os.path.join(ckpt_dir, f"{tag}_last.pth")

    _log(f"Checkpoint (best) : {best_path}")
    _log(f"Checkpoint (last) : {last_path}")

    # CSV log
    csv_path = os.path.join(args.output_dir,
                            f"train_{args.dataset}_{args.model}_{args.defense}.csv")
    csv_f    = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_f)
    csv_w.writerow(["round", "train_acc", "val_acc"])

    best_val  = 0.0
    prev_deltas = None
    t_total   = time.time()

    for rnd in range(1, args.rounds + 1):
        t_rnd = time.time()

        # ── Local training ────────────────────────────────────
        local_deltas = []
        for k in range(args.n_clients):
            # Each client starts from current global model
            local_model = get_model(
                args.model, num_classes=num_classes).to(device)
            local_model.load_state_dict(global_model.state_dict())

            delta = local_train(
                local_model,
                client_datasets[k],
                device,
                local_bs=args.local_bs,
                local_epochs=args.local_epochs,
                lr=args.lr)
            local_deltas.append(delta)

        # ── Apply defense ─────────────────────────────────────
        defended_deltas = apply_defense_to_deltas(
            local_deltas, prev_deltas,
            defense=args.defense,
            alpha=args.alpha,
            sigma=args.sigma,
            device=device)

        # Save current deltas as prev for next round (LCN)
        prev_deltas = local_deltas  # always save true deltas

        # ── FedAvg aggregation ────────────────────────────────
        global_sd = global_model.state_dict()
        for k in global_sd:
            if global_sd[k].dtype.is_floating_point:
                avg_delta = torch.stack(
                    [defended_deltas[i][k].to(device).float()
                     for i in range(args.n_clients)]
                ).mean(dim=0)
                global_sd[k] = global_sd[k].float() + avg_delta
        global_model.load_state_dict(global_sd)

        # ── Evaluate ──────────────────────────────────────────
        val_acc   = evaluate(global_model, test_ds,   device)
        train_acc = evaluate(global_model,
                             Subset(client_datasets[0],
                                    range(min(500, len(client_datasets[0])))),
                             device)

        elapsed = time.time() - t_rnd
        _log(f"Round {rnd:3d}/{args.rounds}  "
             f"train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  "
             f"({elapsed:.1f}s)")
        csv_w.writerow([rnd, f"{train_acc:.4f}", f"{val_acc:.4f}"])
        csv_f.flush()

        # ── Save best checkpoint ──────────────────────────────
        if val_acc > best_val:
            best_val = val_acc
            torch.save(global_model.state_dict(), best_path)
            _log(f"  ★ New best val_acc={best_val:.2f}% — saved to {best_path}")

    # Save last checkpoint
    torch.save(global_model.state_dict(), last_path)
    _log(f"Last checkpoint saved to: {last_path}")

    total_min = (time.time() - t_total) / 60
    _log(f"\nTraining complete in {total_min:.1f} min.  "
         f"Best val_acc={best_val:.2f}%")
    _log(f"Use checkpoint: {best_path}")

    csv_f.close()
    if _log_file:
        _log_file.close()

    return best_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FL Training — generates checkpoint for defense evaluation")

    # Dataset / model
    parser.add_argument("--dataset",       default="cifar10",
                        choices=["cifar10", "tinyimagenet"])
    parser.add_argument("--model",         default="mobilenet",
                        choices=["mobilenet", "resnet18"])

    # FL hyperparameters (match Section V-A setup)
    parser.add_argument("--n_clients",     type=int,   default=5)
    parser.add_argument("--rounds",        type=int,   default=100)
    parser.add_argument("--local_epochs",  type=int,   default=1)
    parser.add_argument("--local_bs",      type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)

    # Defense during training
    parser.add_argument("--defense",       default="none",
                        choices=["none", "dp", "lcn"],
                        help="Defense applied during training "
                             "(use 'none' for baseline checkpoint)")
    parser.add_argument("--alpha",         type=float, default=0.7,
                        help="LCN mixing coefficient (used if defense=lcn)")
    parser.add_argument("--sigma",         type=float, default=1e-3,
                        help="DP noise std (used if defense=dp)")

    # Output
    parser.add_argument("--ckpt_dir",      default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--output_dir",    default="./results",
                        help="Directory for logs and CSV")
    parser.add_argument("--seed",          type=int,   default=42)

    args = parser.parse_args()
    train(args)