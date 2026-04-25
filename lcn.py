"""
lcn.py — LCN defense và DP baseline

LCN noise construction:
    n_k^(t) = beta_k * (1-alpha) * sum_j Dw_j^(t-1)
            = m * beta_k * (1-alpha) * (w^(t) - w^(t-1))

    Ràng buộc duy nhất: sum_k beta_k = 1
    beta_k KHÔNG bị ràng buộc dấu: có thể > 0, = 0, hoặc < 0.

    Hành vi theo giá trị beta_k:
        beta_k > 0 : noise cùng chiều gradient lịch sử
        beta_k = 0 : không thêm noise (client không được bảo vệ)
        beta_k < 0 : noise ngược chiều gradient lịch sử
"""

import torch
import math
import random as _random


def generate_betas(m, seed=None, uniform=False):
    """
    Sinh hệ số beta cho m clients thỏa sum(beta) = 1.
    beta_k KHÔNG bị ràng buộc dấu.
    Mặc định dùng Exponential normalization → Uniform trên simplex (tất cả > 0).
    """
    if uniform:
        return [1.0 / m] * m
    if seed is not None:
        _random.seed(seed)
    xs = [-math.log(_random.random()) for _ in range(m)]
    s  = sum(xs)
    return [x / s for x in xs]


def set_beta_k(beta_k, k, m):
    """
    Tạo vector beta với beta[k] = beta_k được chỉ định,
    các client còn lại chia đều phần còn lại để sum = 1.

    Hỗ trợ beta_k âm, bằng 0 hoặc dương.

    Args:
        beta_k : float, giá trị beta của target client k
        k      : int, index của target client (0-indexed)
        m      : int, số clients
    Returns:
        list of m floats, sum = 1
    """
    remaining = 1.0 - beta_k
    betas = []
    for i in range(m):
        if i == k:
            betas.append(float(beta_k))
        else:
            betas.append(remaining / (m - 1))
    assert abs(sum(betas) - 1.0) < 1e-9, \
        f"sum(betas)={sum(betas):.6f} != 1"
    return betas


def compute_agg_prev(w_cur, w_prev, m):
    """
    Tính sum_j Dw_j^(t-1) = m * (w^(t) - w^(t-1))
    """
    return [m * (wc - wp) for wc, wp in zip(w_cur, w_prev)]


def apply_lcn(true_grad, w_cur, w_prev, alpha, beta_k, m):
    """
    LCN transformation:
        w'_k^(t) = alpha * Dw_k^(t)
                 + beta_k * (1-alpha) * m * (w^(t) - w^(t-1))

    beta_k không bị ràng buộc dấu.
    """
    noise = [beta_k * (1.0 - alpha) * m * (wc - wp)
             for wc, wp in zip(w_cur, w_prev)]
    return [alpha * g + n for g, n in zip(true_grad, noise)]


def apply_dp_noise(true_grad, sigma, device):
    """
    DP baseline: g'_k = g_k + N(0, sigma^2 * I)
    """
    return [
        g + torch.randn_like(g).to(device) * sigma
        for g in true_grad
    ]