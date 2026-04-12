# Defense Capability Evaluation — LCN

Code for **Section V-B: Evaluations on Defense Capability** of the paper
*"Development of a Low-Cost Correlated Noise-based Defense Method against
Gradient Inversion Attacks in Federated Learning"*.

---

## File structure

```
defense_capability/
├── evaluate_defense.py     # Main evaluation loop → CSV
├── visualize.py            # Reconstruction figure generator → PDF/PNG
├── print_latex_tables.py   # CSV → LaTeX table rows
├── attacks.py              # IG and iDLG implementations
├── lcn.py                  # LCN transform + DP noise
├── metrics.py              # PSNR / SSIM / LPIPS
├── models.py               # MobileNet / ResNet18 factory
├── run_all.sh              # Convenience shell script
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Pretrained checkpoints

Put your checkpoints (trained with the FL utility experiments) in:

```
checkpoints/
├── cifar10_mobilenet.pth
└── tinyimagenet_resnet18.pth
```

If no checkpoint is provided, the script runs in **demo mode** with
randomly initialized weights (useful for testing the pipeline).

---

## Usage

### Option A — Run everything at once

```bash
bash run_all.sh             # full run (8000 IG iterations)
bash run_all.sh --quick     # fast test (500 iterations, 3 images)
```

### Option B — Run individual components

**Quantitative evaluation (generates CSV):**
```bash
python evaluate_defense.py \
    --dataset    cifar10 \
    --model      mobilenet \
    --attack     ig \
    --n_samples  10 \
    --checkpoint checkpoints/cifar10_mobilenet.pth \
    --output_dir ./results
```

**Visualization figure:**
```bash
python visualize.py \
    --dataset    cifar10 \
    --model      mobilenet \
    --attack     ig \
    --n_show     4 \
    --ig_iter    8000 \
    --checkpoint checkpoints/cifar10_mobilenet.pth \
    --output_dir ./results/figures
```

**Print LaTeX table rows from CSV:**
```bash
python print_latex_tables.py --results_dir ./results
```

---

## Defense conditions evaluated

| Condition       | Parameter            |
|-----------------|----------------------|
| No Defense      | —                    |
| DP baseline     | σ = 1e-3             |
| DP baseline     | σ = 1e-2             |
| LCN (proposed)  | α = 1.1              |
| LCN (proposed)  | α = 0.9              |
| LCN (proposed)  | α = 0.7              |
| LCN (proposed)  | α = 0.5              |

---

## Metrics

| Metric | Direction for better defense |
|--------|------------------------------|
| PSNR   | ↓ lower is better            |
| SSIM   | ↓ lower is better            |
| LPIPS  | ↑ higher is better           |

---

## Attacks

| Attack | Type               | Reference                    |
|--------|--------------------|------------------------------|
| IG     | Optimization-based | Geiping et al., NeurIPS 2020 |
| iDLG   | Analytics-based    | Zhao et al., arXiv 2020      |

---

## Expected outputs

```
results/
├── defense_capability_cifar10_mobilenet.csv
├── defense_capability_tinyimagenet_resnet18.csv
└── figures/
    ├── fig_defense_ig_cifar10_mobilenet.pdf
    ├── fig_defense_ig_cifar10_mobilenet.png
    ├── fig_defense_idlg_cifar10_mobilenet.pdf
    └── ...
```
