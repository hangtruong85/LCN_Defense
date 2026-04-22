"""
attacks_idlg.py
===============
iDLG — Zhao et al. 2020 "iDLG: Improved Deep Leakage from Gradients"
  - Paper gốc dùng: LeNet (Sigmoid)
  - Mở rộng thêm : MobileNetV2, ResNet18
  - Tham số GIỮ NGUYÊN paper gốc: L-BFGS lr=1.0, 300 iters, L2 grad matching
  - Dùng ground-truth label (không infer)
"""

import time
import os
import csv
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attacks_common import (
    _log, _loss_dir, _compute_dummy_grad, _set_bn_eval,
    set_attack_logger, set_loss_dir
)

def idlg_attack(model, observed_grad, original_image, original_label,
                device,
                # ── Tham số giữ nguyên theo paper gốc ──
                lr=1.0,           # L-BFGS lr=1.0
                n_iter=300,       # 300 iterations
                log_every=30,     # log mỗi 30 iters (giống paper: Iteration/30)
                plot_every=50,    # plot loss curve mỗi 50 iters
                run_tag=""):      # tag để đặt tên file (dataset_model_defense_img)
    """
    iDLG: L-BFGS + L2 gradient matching + closed-form label inference.

    Tham số giữ nguyên theo paper gốc:
      lr=1.0, n_iter=300, optimizer=L-BFGS, loss=L2

    Hỗ trợ model: LeNet (gốc), MobileNetV2, ResNet18 (mở rộng).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    # Dùng ground-truth label (upper bound cho attack strength)
    # Việc infer label là bước phụ của iDLG; dùng GT label giúp đánh giá
    # khả năng reconstruct ảnh một cách công bằng nhất.
    dummy_label = original_label.reshape((1,)).to(device)
    _log(f"        [iDLG] using ground-truth label={dummy_label.item()}")

    dummy     = torch.randn((1, C, H, W), device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy], lr=lr)   # L-BFGS lr=1.0 (paper gốc)

    step       = [0]
    best_loss  = [float("inf")]
    best_dummy = [dummy.detach().clone()]
    loss_hist  = []   # (iter, loss)
    t0         = time.time()

    # CSV writer
    csv_path = None
    csv_f    = None
    csv_w    = None
    if _loss_dir is not None:
        csv_path = os.path.join(_loss_dir, f"loss_idlg_{run_tag}.csv")
        csv_f    = open(csv_path, "w", newline="")
        csv_w    = csv.writer(csv_f)
        csv_w.writerow(["iter", "loss", "best_loss"])

    def _save_loss_plot():
        if _loss_dir is None or len(loss_hist) < 2:
            return
        iters = [x[0] for x in loss_hist]
        losses = [x[1] for x in loss_hist]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(iters, losses, color="#185FA5", linewidth=1.2, label="loss")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Loss (L2)", fontsize=10)
        ax.set_title(f"iDLG loss — {run_tag}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(_loss_dir, f"loss_idlg_{run_tag}.png")
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

    def closure():
        optimizer.zero_grad()
        # L2 gradient matching (đúng paper gốc)
        dummy_grad = _compute_dummy_grad(model, dummy, dummy_label, criterion)
        grad_diff  = sum(
            ((dg - og.to(device)) ** 2).sum()
            for dg, og in zip(dummy_grad, observed_grad)
        )
        grad_diff.backward()

        step[0] += 1
        loss_val = grad_diff.item()
        if loss_val < best_loss[0]:
            best_loss[0]  = loss_val
            best_dummy[0] = dummy.detach().clone()

        loss_hist.append((step[0], loss_val))
        if csv_w is not None:
            csv_w.writerow([step[0], f"{loss_val:.8f}", f"{best_loss[0]:.8f}"])

        if step[0] % log_every == 0 or step[0] == n_iter:
            elapsed = time.time() - t0
            _log(f"        [iDLG] iter {step[0]:3d}/{n_iter}  "
                 f"loss={loss_val:.8f}  best={best_loss[0]:.8f}  "
                 f"elapsed={elapsed:.0f}s")

        if step[0] % plot_every == 0 or step[0] == n_iter:
            _save_loss_plot()

        return grad_diff

    for _ in range(n_iter):
        optimizer.step(closure)

    _log(f"        [iDLG] done  best_loss={best_loss[0]:.8f}")
    return best_dummy[0].squeeze(0).cpu()


# ═════════════════════════════════════════════════════════════
# IG ATTACK
# Geiping et al. 2020 — giữ nguyên tham số paper gốc
# ═════════════════════════════════════════════════════════════