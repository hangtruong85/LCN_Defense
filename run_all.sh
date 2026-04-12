#!/usr/bin/env bash
# run_all.sh — Run the full defense capability evaluation
# All outputs (logs, CSVs, images, figures) go into one timestamped folder.
#
# Usage:
#   bash run_all.sh              # full run
#   bash run_all.sh --quick      # fast test (3 samples, 200 IG iter)

set -e

# ── Parse arguments ───────────────────────────────────────────────
QUICK=0
for arg in "$@"; do
  if [ "$arg" == "--quick" ]; then QUICK=1; fi
done

N_SAMPLES=10
IG_ITER=8000
if [ $QUICK -eq 1 ]; then
  N_SAMPLES=3
  IG_ITER=200
  echo ">>> QUICK MODE: n_samples=$N_SAMPLES, ig_iter=$IG_ITER"
fi

# ── Create ONE timestamped run directory ─────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./results/defense_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo ""
echo "============================================================"
echo " Defense Capability Evaluation"
echo " Run directory: $RUN_DIR"
echo "============================================================"

# Save run config
cat > "$RUN_DIR/run_config.txt" << CONF
timestamp   : $TIMESTAMP
n_samples   : $N_SAMPLES
ig_iter     : $IG_ITER
quick_mode  : $QUICK
datasets    : cifar10  tinyimagenet
models      : mobilenet  resnet18
attacks     : ig  idlg
CONF

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

# ── STEP 1: Quantitative evaluation ──────────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Quantitative evaluation (CSV + per-image PNG)"
echo "============================================================"

#for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10" "tinyimagenet resnet18 $CKPT_TINY"; do
for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10"; do
  read -r DATASET MODEL CKPT <<< "$DATASET_MODEL"
  #for ATTACK in ig idlg; do
  for ATTACK in idlg; do
    echo ""
    echo ">>> Dataset=$DATASET  Model=$MODEL  Attack=$ATTACK"
    python evaluate_defense.py \
      --dataset     "$DATASET" \
      --model       "$MODEL" \
      --attack      "$ATTACK" \
      --n_samples   "$N_SAMPLES" \
      --checkpoint  "$CKPT" \
      --run_dir     "$RUN_DIR" \
      --seed        42
  done
done

# ── STEP 2: Summary figures ───────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Summary figures (PDF + PNG)"
echo "============================================================"

#for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10" "tinyimagenet resnet18 $CKPT_TINY"; do
for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10"; do
  read -r DATASET MODEL CKPT <<< "$DATASET_MODEL"
  #for ATTACK in ig idlg; do
  for ATTACK in idlg; do
    echo ""
    echo ">>> Visualization: Dataset=$DATASET  Model=$MODEL  Attack=$ATTACK"
    python visualize.py \
      --dataset     "$DATASET" \
      --model       "$MODEL" \
      --attack      "$ATTACK" \
      --n_show      4 \
      --ig_iter     "$IG_ITER" \
      --checkpoint  "$CKPT" \
      --output_dir  "$RUN_DIR/figures" \
      --seed        42
  done
done

# ── STEP 3: LaTeX table rows ──────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: LaTeX table rows"
echo "============================================================"
python print_latex_tables.py --results_dir "$RUN_DIR" \
  | tee "$RUN_DIR/latex_tables.txt"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All done. Run directory: $RUN_DIR"
echo ""
echo " Contents:"
find "$RUN_DIR" -type f | sort | sed 's|^|   |'
echo "============================================================"