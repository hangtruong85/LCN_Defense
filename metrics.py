"""
metrics.py — PSNR, SSIM, LPIPS
"""

import numpy as np
import torch


# ─────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────
def compute_psnr(img1_np, img2_np, max_val=1.0):
    """
    Compute PSNR between two HWC float images in [0, 1].
    Returns float (dB). Lower is better for defense.
    """
    mse = np.mean((img1_np.astype(np.float64) -
                   img2_np.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


# ─────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────
def compute_ssim(img1_np, img2_np):
    """
    Compute mean SSIM between two HWC float images in [0, 1].
    Uses skimage if available, otherwise falls back to a manual implementation.
    Lower SSIM is better for defense.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        # skimage expects channel_axis for multichannel
        ssim_val = sk_ssim(img1_np, img2_np,
                           data_range=1.0,
                           channel_axis=-1)
        return float(ssim_val)
    except ImportError:
        return _ssim_manual(img1_np, img2_np)


def _ssim_manual(img1, img2, C1=0.01**2, C2=0.03**2):
    """Pure-numpy SSIM fallback (averaged over channels)."""
    scores = []
    for c in range(img1.shape[2]):
        x = img1[:, :, c].astype(np.float64)
        y = img2[:, :, c].astype(np.float64)
        mu_x = np.mean(x); mu_y = np.mean(y)
        sig_x = np.var(x); sig_y = np.var(y)
        sig_xy = np.mean((x - mu_x) * (y - mu_y))
        num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
        scores.append(num / den)
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────
def compute_lpips(img1_tensor, img2_tensor, lpips_fn, device):
    """
    Compute LPIPS between two CHW tensors (values in arbitrary range,
    will be normalized to [-1, 1] internally).
    Higher LPIPS is better for defense.

    Args:
        img1_tensor : CHW torch.Tensor (original)
        img2_tensor : CHW torch.Tensor (reconstructed)
        lpips_fn    : lpips.LPIPS instance
        device      : torch.device
    Returns:
        float
    """
    def _to_lpips_range(t):
        """Normalize tensor to [-1, 1] for LPIPS."""
        t = t.float()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return t * 2.0 - 1.0

    x = _to_lpips_range(img1_tensor).unsqueeze(0).to(device)
    y = _to_lpips_range(img2_tensor).unsqueeze(0).to(device)

    # LPIPS (AlexNet) cần ảnh ít nhất 32x32
    # MNIST (28x28) hoặc ảnh nhỏ hơn cần được resize
    min_size = 32
    if x.shape[-1] < min_size or x.shape[-2] < min_size:
        import torch.nn.functional as F
        x = F.interpolate(x, size=(min_size, min_size),
                          mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(min_size, min_size),
                          mode="bilinear", align_corners=False)

    # MNIST là grayscale (1 channel) — LPIPS cần 3 channels
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)

    with torch.no_grad():
        val = lpips_fn(x, y)
    return float(val.item())