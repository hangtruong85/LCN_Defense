"""
lcn.py — LCN transform and DP noise utilities
"""

import torch


def apply_lcn(true_grad, prev_grad, alpha):
    """
    LCN transformation (Eq. in paper):
        g'_k^(t) = alpha * g_k^(t) + (1 - alpha) * g_k^(t-1)

    Args:
        true_grad : list of tensors, current round gradient
        prev_grad : list of tensors, previous round gradient
        alpha     : float, mixing coefficient
    Returns:
        list of tensors, transformed gradient
    """
    assert len(true_grad) == len(prev_grad), \
        "true_grad and prev_grad must have same length"
    return [
        alpha * g_cur + (1.0 - alpha) * g_prev
        for g_cur, g_prev in zip(true_grad, prev_grad)
    ]


def apply_dp_noise(true_grad, sigma, device):
    """
    Independent Gaussian noise (DP baseline):
        g'_k^(t) = g_k^(t) + N(0, sigma^2 * I)

    Args:
        true_grad : list of tensors
        sigma     : float, noise standard deviation
        device    : torch.device
    Returns:
        list of tensors, perturbed gradient
    """
    return [
        g + torch.randn_like(g).to(device) * sigma
        for g in true_grad
    ]
