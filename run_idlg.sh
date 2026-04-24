#!/usr/bin/env bash
# run_idlg.sh — Defense capability evaluation với iDLG attack
#
# Model    : lenet, mobilenet, resnet18
# Attack   : iDLG (Zhao et al. 2020)
#            L-BFGS, ground-truth label, L2 gradient matching
#
# Usage:
#   bash run_idlg.sh                     # full run (lr=0.1, iter=500)
#   bash run_idlg.sh --quick             # test nhanh (3 ảnh, iter=300)
#   bash run_idlg.sh --lr 0.01           # tùy chỉnh learning rate
#   bash run_idlg.sh --iter 1000         # tùy chỉnh số iterations
#   bash run_idlg.sh --lr 0.01 --iter 1000 --quick

set -e

QUICK=0
IDLG_LR=0.1
IDLG_ITER=500
N_SAMPLES=10

# ── Parse arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick) QUICK=1;        shift ;;
    --lr)    IDLG_LR="$2";   shift 2 ;;
    --iter)  IDLG_ITER="$2"; shift 2 ;;
    *)       shift ;;
  esac
done

if [ $QUICK -eq 1 ]; then
  N_SAMPLES=3
  IDLG_ITER=300
  echo ">>> QUICK MODE: n_samples=$N_SAMPLES  idlg_iter=$IDLG_ITER  lr=$IDLG_LR"
fi

# ── Timestamped run directory ──────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./results/idlg_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo ""
echo "============================================================"
echo " iDLG Attack — Defense Capability Evaluation"
echo " Run directory : $RUN_DIR"
echo " L-BFGS lr=$IDLG_LR  iter=$IDLG_ITER  label=ground-truth"
echo "============================================================"

cat > "$RUN_DIR/run_config.txt" << CONF
attack       : idlg
timestamp    : $TIMESTAMP
n_samples    : $N_SAMPLES
idlg_iter    : $IDLG_ITER
idlg_lr      : $IDLG_LR
quick_mode   : $QUICK
CONF

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

# LeNet (iDLG paper gốc) — không cần checkpoint, dùng random weights
CONFIGS_LENET=(
  #"mnist     lenet  none"
  #"cifar10   lenet  none"
  #"cifar100  lenet  none"
  "lfw       lenet  none"
)

# MobileNet / ResNet18 — cần checkpoint đã train
CONFIGS_PRETRAINED=(
  "cifar10      mobilenet $CKPT_CIFAR10"
  "tinyimagenet resnet18  $CKPT_TINY"
)

# Gộp tất cả: LeNet trước, sau đó pretrained
CONFIGS=(
  "${CONFIGS_LENET[@]}"
  "${CONFIGS_PRETRAINED[@]}"
)

# ── STEP 1: Quantitative evaluation ───────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Quantitative evaluation (CSV + per-image PNG)"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
  read -r DATASET MODEL CKPT <<< "$CFG"
  CKPT_ARG=""
  if [ "$CKPT" != "none" ] && [ -f "$CKPT" ]; then
    CKPT_ARG="--checkpoint $CKPT"
  fi
  echo ""
  echo ">>> Dataset=$DATASET  Model=$MODEL  Attack=idlg  lr=$IDLG_LR  iter=$IDLG_ITER"
  python evaluate_defense_idlg.py \
    --dataset    "$DATASET" \
    --model      "$MODEL" \
    --attack     idlg \
    --n_samples  "$N_SAMPLES" \
    --idlg_iter  "$IDLG_ITER" \
    --idlg_lr    "$IDLG_LR" \
    $CKPT_ARG \
    --run_dir    "$RUN_DIR" \
    --seed       42
done

# ── STEP 2: Summary figures ────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Summary figures (ghép từ ảnh đã có)"
echo "============================================================"

python make_figures.py \
  --run_dir "$RUN_DIR" \
  --n_show  5
  python make_figures.py \
  --run_dir "results/idlg_20260422_132132" \
  --n_show  5

# ── STEP 3: LaTeX tables ───────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: LaTeX table rows"
echo "============================================================"
python print_latex_tables.py --results_dir "$RUN_DIR" \
  | tee "$RUN_DIR/latex_tables.txt"

echo ""
echo "============================================================"
echo " All done. Run directory: $RUN_DIR"
echo " Contents:"
find "$RUN_DIR" -type f | sort | sed 's|^|   |'
echo "============================================================"