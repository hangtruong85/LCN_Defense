#!/usr/bin/env bash
# run_ig.sh — Defense capability evaluation với IG attack
#
# Model    : lenet, mobilenet, resnet18
# Attack   : IG (Geiping et al. 2020)
#            Adam lr=0.1, 24000 iter, 8 restarts, cosine+TV, signed, boxed
#
# Usage:
#   bash run_ig.sh              # full run
#   bash run_ig.sh --quick      # test nhanh (3 ảnh, 1000 iter, 1 restart)

set -e

QUICK=0
for arg in "$@"; do
  if [ "$arg" == "--quick" ]; then QUICK=1; fi
done

# ── Tham số IG (đúng theo paper gốc) ──────────────────────────
N_SAMPLES=1
IG_ITER=8000
IG_RESTARTS=8

if [ $QUICK -eq 1 ]; then
  N_SAMPLES=3
  IG_ITER=1000
  IG_RESTARTS=1
  echo ">>> QUICK MODE: n_samples=$N_SAMPLES  ig_iter=$IG_ITER  restarts=$IG_RESTARTS"
fi

# ── Timestamped run directory ──────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./results/ig_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo ""
echo "============================================================"
echo " IG Attack — Defense Capability Evaluation"
echo " Run directory : $RUN_DIR"
echo " Adam lr=0.1  iter=$IG_ITER  restarts=$IG_RESTARTS"
echo "============================================================"

cat > "$RUN_DIR/run_config.txt" << CONF
attack       : ig
timestamp    : $TIMESTAMP
n_samples    : $N_SAMPLES
ig_iter      : $IG_ITER
ig_restarts  : $IG_RESTARTS
quick_mode   : $QUICK
CONF

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

CONFIGS=(
  #"cifar10      lenet     none"
  #"cifar10      mobilenet $CKPT_CIFAR10"
  #"cifar10      resnet18  none"
  #"tinyimagenet mobilenet none"
  "tinyimagenet resnet18  $CKPT_TINY"
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
  echo ">>> Dataset=$DATASET  Model=$MODEL  Attack=ig"
  python evaluate_defense_ig.py \
    --dataset   "$DATASET" \
    --model     "$MODEL" \
    --attack    ig \
    --n_samples "$N_SAMPLES" \
    $CKPT_ARG \
    --run_dir   "$RUN_DIR" \
    --seed      52
done

# ── STEP 2: Summary figures ────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Summary figures (ghép từ ảnh đã có)"
echo "============================================================"

python make_figures.py \
  --run_dir "$RUN_DIR" \
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