"""
attacks.py
==========
Triển khai iDLG và IG đúng theo paper gốc, với mở rộng model.

iDLG — Zhao et al. 2020 "iDLG: Improved Deep Leakage from Gradients"
  - Paper gốc dùng: LeNet (Sigmoid)
  - Mở rộng thêm : MobileNetV2, ResNet18
  - Tham số giữ nguyên: L-BFGS lr=1.0, 300 iters, L2 grad matching

IG — Geiping et al. 2020 "Inverting Gradients"
  - Paper gốc dùng: ResNet18 (untrained/pretrained)
  - Mở rộng thêm : MobileNetV2, LeNet
  - Tham số giữ nguyên: Adam lr=0.1, 24000 iters, 8 restarts,
                         cosine loss, TV reg, signed, boxed, lr_decay
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
_attack_log_file = None

def set_attack_logger(f):
    global _attack_log_file
    _attack_log_file = f

def _log(msg):
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _attack_log_file is not None:
        _attack_log_file.write(line + "\n")
        _attack_log_file.flush()


# ─────────────────────────────────────────────────────────────
# LeNet — paper gốc iDLG
# ─────────────────────────────────────────────────────────────
class LeNet(nn.Module):
    """
    Đúng theo iDLG paper gốc:
      - Sigmoid activation (không phải ReLU)
      - 3 conv layers + 1 FC
      - hidden=768 cho ảnh 32x32x3
    """
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def weights_init_idlg(m):
    """Weight init đúng theo iDLG paper gốc: Uniform(-0.5, 0.5)."""
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# BatchNorm fix cho batch_size=1
# ─────────────────────────────────────────────────────────────
def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────────────────────
# Compute gradient từ dummy input
# ─────────────────────────────────────────────────────────────
def _compute_dummy_grad(model, dummy_data, dummy_label, criterion):
    model.train()
    _set_bn_eval(model)
    model.zero_grad()
    out  = model(dummy_data)
    loss = criterion(out, dummy_label)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return list(grad)


# ─────────────────────────────────────────────────────────────
# TV regularizer
# ─────────────────────────────────────────────────────────────
def _total_variation(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


# ─────────────────────────────────────────────────────────────
# Label inference — ĐÚNG theo iDLG paper gốc
# label_pred = argmin( sum(grad[-2], dim=-1) )
# sum theo dim=-1 (chiều features), argmin theo chiều classes
# ─────────────────────────────────────────────────────────────
def _infer_label(observed_grad, model):
    named = list(model.named_parameters())
    assert len(named) == len(observed_grad), \
        f"Param mismatch: model={len(named)} grad={len(observed_grad)}"

    # Tìm FC weight layer cuối (2-D tensor)
    fc_grad = None
    fc_name = "unknown"
    for i in range(len(named) - 1, -1, -1):
        name, param = named[i]
        if param.dim() == 2 and "weight" in name:
            fc_grad  = observed_grad[i]
            fc_name  = name
            break

    if fc_grad is None:
        fc_grad = observed_grad[-2]
        fc_name = "fallback[-2]"

    # Paper gốc: argmin(sum(grad[-2], dim=-1))
    label_pred = torch.argmin(
        torch.sum(fc_grad, dim=-1), dim=-1
    ).detach().reshape((1,))

    _log(f"        [label] layer='{fc_name}'  shape={list(fc_grad.shape)}  "
         f"inferred={label_pred.item()}")
    return label_pred


# ═════════════════════════════════════════════════════════════
# iDLG ATTACK
# Zhao et al. 2020 — giữ nguyên tham số paper gốc
# ═════════════════════════════════════════════════════════════
def idlg_attack(model, observed_grad, original_image, original_label,
                device,
                # ── Tham số giữ nguyên theo paper gốc ──
                lr=1.0,           # L-BFGS lr=1.0
                n_iter=300,       # 300 iterations
                log_every=30):    # log mỗi 30 iters (giống paper: Iteration/30)
    """
    iDLG: L-BFGS + L2 gradient matching + closed-form label inference.

    Tham số giữ nguyên theo paper gốc:
      lr=1.0, n_iter=300, optimizer=L-BFGS, loss=L2

    Hỗ trợ model: LeNet (gốc), MobileNetV2, ResNet18 (mở rộng).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    # Label inference (đúng paper gốc)
    label_pred    = _infer_label(observed_grad, model)
    label_correct = (label_pred.item() == original_label.item())
    _log(f"        [iDLG] label {'CORRECT' if label_correct else 'WRONG'}  "
         f"inferred={label_pred.item()}  true={original_label.item()}")

    dummy       = torch.randn((1, C, H, W), device=device).requires_grad_(True)
    optimizer   = torch.optim.LBFGS([dummy], lr=lr)   # L-BFGS lr=1.0 (paper gốc)
    dummy_label = label_pred.to(device)

    step       = [0]
    best_loss  = [float("inf")]
    best_dummy = [dummy.detach().clone()]
    t0         = time.time()

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

        if step[0] % log_every == 0 or step[0] == n_iter:
            elapsed = time.time() - t0
            _log(f"        [iDLG] iter {step[0]:3d}/{n_iter}  "
                 f"loss={loss_val:.8f}  best={best_loss[0]:.8f}  "
                 f"elapsed={elapsed:.0f}s")
        return grad_diff

    for _ in range(n_iter):
        optimizer.step(closure)

    _log(f"        [iDLG] done  best_loss={best_loss[0]:.8f}  "
         f"label_correct={label_correct}")
    return best_dummy[0].squeeze(0).cpu()


# ═════════════════════════════════════════════════════════════
# IG ATTACK
# Geiping et al. 2020 — giữ nguyên tham số paper gốc
# ═════════════════════════════════════════════════════════════
def ig_attack(model, observed_grad, original_image, device,
              # ── Tham số giữ nguyên theo paper gốc ──
              n_iter=24_000,     # max_iterations=24000
              lr=0.1,            # lr=0.1
              tv_weight=1e-4,    # total_variation (paper dùng 1e-4 cho cifar)
              n_restarts=8,      # restarts=8
              signed=True,       # signed=True
              boxed=True,        # boxed=True
              lr_decay=True,     # lr_decay=True
              log_every=500):
    """
    IG: cosine similarity + TV + Adam + lr_decay + restarts + signed + boxed.

    Tham số giữ nguyên theo paper gốc:
      lr=0.1, n_iter=24000, restarts=8, TV, signed, boxed, lr_decay

    Hỗ trợ model: ResNet18 (gốc), MobileNetV2, LeNet (mở rộng).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    # Label inference dùng chung
    label_pred  = _infer_label(observed_grad, model)
    dummy_label = label_pred.to(device)

    best_rec   = None
    best_score = float("inf")   # scoring = cosine loss (thấp hơn = tốt hơn)

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

            # boxed: clamp to [0,1] normalized range (đúng paper gốc)
            if boxed:
                with torch.no_grad():
                    dummy.clamp_(-1.0, 1.0)

            cos_loss_val = cos_loss.item()

            if i % log_every == 0 or i == n_iter:
                elapsed = time.time() - t_restart
                eta     = elapsed / i * (n_iter - i)
                _log(f"        [IG] restart {restart+1}  "
                     f"iter {i:5d}/{n_iter}  "
                     f"cos={cos_loss_val:.4f}  tv={tv_loss.item():.4f}  "
                     f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # Scoring: lấy restart cho cosine loss thấp nhất (paper gốc: scoring_choice=loss)
        if cos_loss_val < best_score:
            best_score = cos_loss_val
            best_rec   = dummy.squeeze(0).detach().cpu()
            _log(f"        [IG] restart {restart+1} → new best  "
                 f"cos_loss={best_score:.4f}")
        else:
            _log(f"        [IG] restart {restart+1} → not better  "
                 f"cos_loss={cos_loss_val:.4f}  best={best_score:.4f}")

    _log(f"        [IG] all restarts done  best_cos={best_score:.4f}")
    return best_rec