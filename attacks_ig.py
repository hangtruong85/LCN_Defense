"""
attacks_ig.py
=============
IG — Geiping et al. 2020 "Inverting Gradients"
  - Paper gốc dùng: ResNet18 (untrained/pretrained)
  - Mở rộng thêm : MobileNetV2, LeNet
  - Tham số GIỮ NGUYÊN paper gốc: Adam lr=0.1, 24000 iters, 8 restarts,
                                   cosine loss, TV reg, signed, boxed, lr_decay

Fixes so với attacks.py gốc:
  - boxed clamp: [-1,1] → [-3,3] để phù hợp với normalized pixel range
    của TinyImageNet (valid range ≈ [-2.1, 2.2]), không ảnh hưởng iDLG
"""

import time
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attacks_common import (
    _log, _loss_dir, _compute_dummy_grad, _total_variation,
    _infer_label, set_attack_logger, set_loss_dir
)

def ig_attack(model, observed_grad, original_image, device,
              original_label=None,  # ground-truth label
              # ── Tham số giữ nguyên theo paper gốc ──
              n_iter=24_000,     # max_iterations=24000
              lr=0.1,            # lr=0.1
              tv_weight=1e-4,    # total_variation (paper dùng 1e-4 cho cifar)
              n_restarts=8,      # restarts=8
              signed=True,       # signed=True
              boxed=True,        # boxed=True
              lr_decay=True,     # lr_decay=True
              log_every=500,
              plot_every=50,     # plot loss curve mỗi 50 iters
              run_tag=""):       # tag để đặt tên file
    """
    IG: cosine similarity + TV + Adam + lr_decay + restarts + signed + boxed.

    Tham số giữ nguyên theo paper gốc:
      lr=0.1, n_iter=24000, restarts=8, TV, signed, boxed, lr_decay

    Hỗ trợ model: ResNet18 (gốc), MobileNetV2, LeNet (mở rộng).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    # Dùng ground-truth label nếu có, fallback về inference
    if original_label is not None:
        dummy_label = original_label.reshape((1,)).to(device)
        _log(f"        [IG] using ground-truth label={dummy_label.item()}")
    else:
        label_pred  = _infer_label(observed_grad, model)
        dummy_label = label_pred.to(device)

    best_rec   = None
    best_score = float("inf")
    loss_hist  = []   # (restart, iter, cos_loss)

    # CSV
    csv_path = None
    csv_f    = None
    csv_w    = None
    if _loss_dir is not None:
        csv_path = os.path.join(_loss_dir, f"loss_ig_{run_tag}.csv")
        csv_f    = open(csv_path, "w", newline="")
        csv_w    = csv.writer(csv_f)
        csv_w.writerow(["restart", "iter", "cos_loss", "tv_loss"])

    def _save_ig_plot():
        if _loss_dir is None or len(loss_hist) < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 3))
        for r in sorted(set(x[0] for x in loss_hist)):
            sub = [(x[1], x[2]) for x in loss_hist if x[0] == r]
            ax.plot([x[0] for x in sub], [x[1] for x in sub],
                    linewidth=1.0, label=f"restart {r+1}")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Cosine loss", fontsize=10)
        ax.set_title(f"IG loss — {run_tag}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(_loss_dir, f"loss_ig_{run_tag}.png")
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

    for restart in range(n_restarts):
        _log(f"        [IG] restart {restart+1}/{n_restarts}  "
             f"n_iter={n_iter}")
        t_restart = time.time()

        # Init: randn (đúng paper gốc)
        dummy     = torch.randn((1, C, H, W), device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy], lr=lr)

        # lr_decay: MultiStep tại 1/3 và 2/3 (đúng paper gốc)
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[n_iter // 3, 2 * n_iter // 3],
                gamma=0.1)
        else:
            scheduler = None

        cos_loss_val = float("inf")

        for i in range(1, n_iter + 1):
            optimizer.zero_grad()

            dummy_grad = _compute_dummy_grad(
                model, dummy, dummy_label, criterion)

            # Cosine similarity loss (đúng paper gốc)
            # signed: dùng gradient có dấu (signed=True trong paper)
            if signed:
                obs  = [og.to(device).sign() for og in observed_grad]
                dum  = [dg.sign()            for dg in dummy_grad]
            else:
                obs  = [og.to(device) for og in observed_grad]
                dum  = dummy_grad

            cos_loss = 1.0 - sum(
                F.cosine_similarity(d.flatten(), o.flatten(), dim=0)
                for d, o in zip(dum, obs)
            ) / len(dum)

            # TV regularization
            tv_loss  = _total_variation(dummy)
            total    = cos_loss + tv_weight * tv_loss
            total.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            # boxed: clamp theo range hợp lệ của normalized pixels
            # Với ImageNet normalization: pixel ∈ [0,1] trước norm
            # → normalized range ≈ [(0-mean)/std, (1-mean)/std]
            # Dùng [-3, 3] là safe bound bao phủ mọi dataset thông dụng
            if boxed:
                with torch.no_grad():
                    dummy.clamp_(-3.0, 3.0)

            cos_loss_val = cos_loss.item()

            loss_hist.append((restart, i, cos_loss_val))
            if csv_w is not None:
                csv_w.writerow([restart+1, i,
                                 f"{cos_loss_val:.6f}",
                                 f"{tv_loss.item():.6f}"])

            if i % log_every == 0 or i == n_iter:
                elapsed = time.time() - t_restart
                eta     = elapsed / i * (n_iter - i)
                _log(f"        [IG] restart {restart+1}  "
                     f"iter {i:5d}/{n_iter}  "
                     f"cos={cos_loss_val:.4f}  tv={tv_loss.item():.4f}  "
                     f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

            if i % plot_every == 0 or i == n_iter:
                _save_ig_plot()

        # Scoring: lấy restart cho cosine loss thấp nhất (paper gốc: scoring_choice=loss)
        if cos_loss_val < best_score:
            best_score = cos_loss_val
            best_rec   = dummy.squeeze(0).detach().cpu()
            _log(f"        [IG] restart {restart+1} → new best  "
                 f"cos_loss={best_score:.4f}")
        else:
            _log(f"        [IG] restart {restart+1} → not better  "
                 f"cos_loss={cos_loss_val:.4f}  best={best_score:.4f}")

    if csv_f is not None:
        csv_f.close()
    _save_ig_plot()
    _log(f"        [IG] all restarts done  best_cos={best_score:.4f}")
    return best_rec