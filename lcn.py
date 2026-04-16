"""
lcn.py — LCN defense và DP baseline

LCN (phương pháp đề xuất mới):
    Tại round t, client k upload:
        w'_k^(t) = alpha * Dw_k^(t) + n_k^(t)

    Nhiễu được thiết kế:
        n_k^(t) = beta_k * (1 - alpha) * sum_j Dw_j^(t-1)
                = m * beta_k * (1 - alpha) * (w^(t) - w^(t-1))

    Trong đó:
        - beta_k  : hệ số riêng của client k, do trusted third party
                    phân phối một lần trước khi training
        - sum(beta_k) = 1
        - w^(t), w^(t-1): hai global model liên tiếp, client đã có sẵn

    Chứng minh: sum_k n_k^(t) = (1-alpha) * sum_j Dw_j^(t-1)
                               ≈ (1-alpha) * sum_j Dw_j^(t)
    nên sum_k w'_k^(t) ≈ sum_k Dw_k^(t) → model utility preserved.
"""

import torch


# ─────────────────────────────────────────────────────────────────
# LCN — phương pháp mới
# ─────────────────────────────────────────────────────────────────

def generate_betas(m, seed=None):
    """
    Tạo hệ số beta cho m clients thỏa sum(beta) = 1, beta_k > 0.
    Trong thực tế được phân phối bởi trusted third party.
    Ở đây dùng phân phối đều: beta_k = 1/m cho tất cả k.

    Args:
        m    : số clients
        seed : random seed (None = đều nhau)
    Returns:
        list of m floats, sum = 1
    """
    if seed is not None:
        torch.manual_seed(seed)
        raw  = torch.rand(m)
        betas = (raw / raw.sum()).tolist()
    else:
        betas = [1.0 / m] * m
    return betas


def compute_agg_prev(w_cur, w_prev, m):
    """
    Tính tổng gradient vòng trước từ hai global model liên tiếp:
        sum_j Dw_j^(t-1) = m * (w^(t) - w^(t-1))

    Args:
        w_cur  : list of tensors, global model w^(t)
        w_prev : list of tensors, global model w^(t-1)
        m      : số clients
    Returns:
        list of tensors, sum_j Dw_j^(t-1)
    """
    return [m * (wc - wp) for wc, wp in zip(w_cur, w_prev)]


def apply_lcn(true_grad, w_cur, w_prev, alpha, beta_k, m):
    """
    LCN transformation (phương pháp mới):
        w'_k^(t) = alpha * Dw_k^(t)
                 + beta_k * (1-alpha) * m * (w^(t) - w^(t-1))

    Args:
        true_grad : list of tensors, gradient thực Dw_k^(t)
        w_cur     : list of tensors, global model w^(t)
                    (dạng list of tensors, cùng cấu trúc true_grad)
        w_prev    : list of tensors, global model w^(t-1)
        alpha     : float, mixing coefficient
        beta_k    : float, hệ số riêng của client k
        m         : int, số clients
    Returns:
        list of tensors, perturbed gradient w'_k^(t)
    """
    assert len(true_grad) == len(w_cur) == len(w_prev), \
        "true_grad, w_cur, w_prev must have same length"

    # n_k^(t) = beta_k * (1-alpha) * m * (w^(t) - w^(t-1))
    noise = [beta_k * (1.0 - alpha) * m * (wc - wp)
             for wc, wp in zip(w_cur, w_prev)]

    return [alpha * g + n for g, n in zip(true_grad, noise)]


# ─────────────────────────────────────────────────────────────────
# DP baseline — giữ nguyên
# ─────────────────────────────────────────────────────────────────

def apply_dp_noise(true_grad, sigma, device):
    """
    Independent Gaussian noise (DP baseline):
        g'_k^(t) = g_k^(t) + N(0, sigma^2 * I)
    """
    return [
        g + torch.randn_like(g).to(device) * sigma
        for g in true_grad
    ]