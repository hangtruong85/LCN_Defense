"""
attacks_common.py
=================
Shared utilities dùng cho cả iDLG và IG.
"""

import time
import os
import csv
import torch
import torch.nn as nn
import torchvision.models as tv_models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────
_attack_log_file = None
_loss_dir        = None

def set_loss_dir(d):
    global _loss_dir
    _loss_dir = d
    os.makedirs(d, exist_ok=True)

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
# Label inference — đúng theo iDLG paper gốc
# ─────────────────────────────────────────────────────────────
def _infer_label(observed_grad, model):
    named = list(model.named_parameters())
    assert len(named) == len(observed_grad), \
        f"Param mismatch: model={len(named)} grad={len(observed_grad)}"

    fc_grad = None
    fc_name = "unknown"
    for i in range(len(named) - 1, -1, -1):
        name, param = named[i]
        if param.dim() == 2 and "weight" in name:
            fc_grad = observed_grad[i]
            fc_name = name
            break

    if fc_grad is None:
        fc_grad = observed_grad[-2]
        fc_name = "fallback[-2]"

    label_pred = torch.argmin(
        torch.sum(fc_grad, dim=-1), dim=-1
    ).detach().reshape((1,))

    _log(f"        [label] layer='{fc_name}'  shape={list(fc_grad.shape)}  "
         f"inferred={label_pred.item()}")
    return label_pred