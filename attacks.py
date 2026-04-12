"""
attacks.py — IG and iDLG attack implementations

IG  : Geiping et al., "Inverting Gradients", NeurIPS 2020
iDLG: Zhao et al., "iDLG: Improved Deep Leakage from Gradients", 2020
"""

import time
import torch
import torch.nn as nn

# Shared log handle — set by evaluate_defense via set_attack_logger()
_attack_log_file = None

def set_attack_logger(f):
    """Pass the open log file handle from evaluate_defense."""
    global _attack_log_file
    _attack_log_file = f

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _attack_log_file is not None:
        _attack_log_file.write(line + "\n")
        _attack_log_file.flush()


# ─────────────────────────────────────────────────────────────
# BatchNorm fix
# ─────────────────────────────────────────────────────────────
def _set_bn_eval(model):
    """Keep BN layers in eval mode to support batch_size=1."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────────────────────
# Shared utility: compute gradient from dummy input
# ─────────────────────────────────────────────────────────────
def _compute_dummy_grad(model, dummy_data, dummy_label, criterion):
    """
    Return list-of-tensors gradient induced by (dummy_data, dummy_label).
    BN layers are forced to eval mode so batch_size=1 is safe.
    """
    model.train()
    _set_bn_eval(model)
    model.zero_grad()
    out  = model(dummy_data)
    loss = criterion(out, dummy_label)
    grad = torch.autograd.grad(loss, model.parameters(),
                                create_graph=True)
    return list(grad)


def _total_variation(x):
    """Anisotropic TV regularizer for 4-D tensor (N, C, H, W)."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


# ─────────────────────────────────────────────────────────────
# IG Attack  (cosine-similarity matching + TV prior)
# ─────────────────────────────────────────────────────────────
def ig_attack(model, observed_grad, original_image, device,
              n_iter=500, lr=0.1, tv_weight=1e-4,
              n_restarts=1, log_every=50):
    """
    Reconstruct private image via gradient inversion (IG).

    Args:
        model          : the global model (eval mode, weights fixed)
        observed_grad  : list of tensors — the (possibly defended) gradient
        original_image : CHW tensor — used only to determine shape/device
        device         : torch.device
        n_iter         : optimization iterations
        lr             : learning rate for dummy data
        tv_weight      : weight of total-variation regularizer
        n_restarts     : number of random restarts (best kept)
        log_every      : print progress every N iterations

    Returns:
        best reconstructed image as CHW tensor (detached, cpu)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    label_pred = _infer_label(observed_grad, model)

    best_rec  = None
    best_loss = float("inf")

    for restart in range(n_restarts):
        _log(f"        [IG] restart {restart+1}/{n_restarts} — "
              f"{n_iter} iterations ...")
        t_restart = time.time()

        dummy = torch.randn((1, C, H, W), device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy], lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[n_iter // 3, 2 * n_iter // 3],
            gamma=0.1)

        dummy_label = label_pred.to(device)
        cos_loss = torch.tensor(0.0)

        for i in range(1, n_iter + 1):
            optimizer.zero_grad()

            dummy_grad = _compute_dummy_grad(
                model, dummy, dummy_label, criterion)

            cos_loss = 1.0 - sum(
                torch.nn.functional.cosine_similarity(
                    dg.flatten(), og.flatten().to(device), dim=0)
                for dg, og in zip(dummy_grad, observed_grad)
            ) / len(dummy_grad)

            tv_loss = _total_variation(dummy)
            total   = cos_loss + tv_weight * tv_loss

            total.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                dummy.clamp_(-2.5, 2.5)

            # Progress log every log_every iterations
            if i % log_every == 0 or i == n_iter:
                elapsed = time.time() - t_restart
                eta     = elapsed / i * (n_iter - i)
                _log(f"        [IG] iter {i:5d}/{n_iter}  "
                      f"loss={cos_loss.item():.4f}  "
                      f"elapsed={elapsed:.0f}s  "
                      f"ETA={eta:.0f}s")

        final_loss = cos_loss.item()
        _log(f"        [IG] restart {restart+1} done — "
              f"final loss={final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_rec  = dummy.squeeze(0).detach().cpu()

    return best_rec


# ─────────────────────────────────────────────────────────────
# iDLG Attack  (closed-form label + L2 gradient matching)
# ─────────────────────────────────────────────────────────────
def idlg_attack(model, observed_grad, original_image, original_label,
                device, n_iter=300, lr=0.1, log_every=50):
    """
    Reconstruct private image via iDLG.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    C, H, W   = original_image.shape

    label_pred = _infer_label(observed_grad, model)
    _log(f"        [iDLG] inferred label={label_pred.item()}  "
          f"true label={original_label.item()}")

    dummy     = torch.randn((1, C, H, W), device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy], lr=lr)
    dummy_label = label_pred.to(device)

    step = [0]

    def closure():
        optimizer.zero_grad()
        dummy_grad = _compute_dummy_grad(
            model, dummy, dummy_label, criterion)
        loss = sum(
            torch.norm(dg - og.to(device)) ** 2
            for dg, og in zip(dummy_grad, observed_grad)
        )
        loss.backward()

        step[0] += 1
        if step[0] % log_every == 0 or step[0] == n_iter:
            _log(f"        [iDLG] step {step[0]:3d}/{n_iter}  "
                  f"loss={loss.item():.4f}")
        return loss

    for _ in range(n_iter):
        optimizer.step(closure)
        with torch.no_grad():
            dummy.clamp_(-2.5, 2.5)

    return dummy.squeeze(0).detach().cpu()


# ─────────────────────────────────────────────────────────────
# Label inference
# ─────────────────────────────────────────────────────────────
def _infer_label(observed_grad, model):
    """
    Infer ground-truth label from the gradient of the output layer.
    """
    last_grad  = observed_grad[-2]
    label_pred = torch.argmin(last_grad.sum(dim=1)).reshape((1,))
    return label_pred