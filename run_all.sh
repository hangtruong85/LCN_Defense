#!/usr/bin/env bash
# run_all.sh — Defense capability evaluation
#
# Models   : lenet (paper gốc iDLG), mobilenet, resnet18
# Attacks  : idlg (L-BFGS 500 iter, lr=0.1  ← điều chỉnh)
#            ig   (Adam 24000 iter, lr=0.1, 8 restarts, cosine+TV)
#
# Usage:
#   bash run_all.sh              # full run
#   bash run_all.sh --quick      # test nhanh (3 ảnh, 1000 IG iter, 1 restart)

set -e

QUICK=0
for arg in "$@"; do
  if [ "$arg" == "--quick" ]; then QUICK=1; fi
done

# ── Tham số mặc định đúng theo paper gốc ─────────────────────
N_SAMPLES=10
IG_ITER=24000        # IG paper gốc: max_iterations=24000
IG_RESTARTS=8        # IG paper gốc: restarts=8
IDLG_ITER=500        # tăng từ 300 → 500
IDLG_LR=0.1          # giảm từ 1.0 → 0.1

if [ $QUICK -eq 1 ]; then
  N_SAMPLES=3
  IG_ITER=1000
  IG_RESTARTS=1
  IDLG_ITER=300
  IDLG_LR=0.1
  echo ">>> QUICK MODE: n_samples=$N_SAMPLES  ig_iter=$IG_ITER  ig_restarts=$IG_RESTARTS"
fi

# ── Timestamped run directory ─────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./results/defense_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo ""
echo "============================================================"
echo " Defense Capability Evaluation"
echo " Run directory : $RUN_DIR"
echo " IG            : Adam lr=0.1, iter=$IG_ITER, restarts=$IG_RESTARTS"
echo " iDLG          : L-BFGS lr=$IDLG_LR, iter=$IDLG_ITER"
echo "============================================================"

cat > "$RUN_DIR/run_config.txt" << CONF
timestamp    : $TIMESTAMP
n_samples    : $N_SAMPLES
ig_iter      : $IG_ITER
ig_restarts  : $IG_RESTARTS
idlg_iter    : $IDLG_ITER
idlg_lr      : $IDLG_LR
quick_mode   : $QUICK
CONF

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

# ── STEP 1: Quantitative evaluation ──────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Quantitative evaluation"
echo "============================================================"

# Dataset-model-checkpoint combinations
# Format: "dataset model checkpoint"
CONFIGS=(
  "tinyimagenet lenet     none"
  "cifar10 lenet     none"
  #"cifar10 mobilenet $CKPT_CIFAR10"
  #"cifar10 resnet18  none"
  #"tinyimagenet mobilenet none"
  #"tinyimagenet resnet18  $CKPT_TINY"
)

for CFG in "${CONFIGS[@]}"; do
  read -r DATASET MODEL CKPT <<< "$CFG"

  # lenet và resnet18 trên cifar10 không có checkpoint → random weights (demo)
  # nhưng vẫn chạy được để test pipeline

  CKPT_ARG=""
  if [ "$CKPT" != "none" ] && [ -f "$CKPT" ]; then
    CKPT_ARG="--checkpoint $CKPT"
  fi

  for ATTACK in idlg; do
    echo ""
    echo ">>> Dataset=$DATASET  Model=$MODEL  Attack=$ATTACK"
    python evaluate_defense.py \
      --dataset    "$DATASET" \
      --model      "$MODEL" \
      --attack     "$ATTACK" \
      --n_samples  "$N_SAMPLES" \
      --idlg_iter  "$IDLG_ITER" \
      --idlg_lr    "$IDLG_LR" \
      $CKPT_ARG \
      --run_dir    "$RUN_DIR" \
      --seed       42
  done
done

# ── STEP 2: Summary figures ───────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Summary figures"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
  read -r DATASET MODEL CKPT <<< "$CFG"

  CKPT_ARG=""
  if [ "$CKPT" != "none" ] && [ -f "$CKPT" ]; then
    CKPT_ARG="--checkpoint $CKPT"
  fi

  for ATTACK in idlg; do
    echo ""
    echo ">>> Visualization: Dataset=$DATASET  Model=$MODEL  Attack=$ATTACK"
    python visualize.py \
      --dataset    "$DATASET" \
      --model      "$MODEL" \
      --attack     "$ATTACK" \
      --n_show     4 \
      --ig_iter    "$IG_ITER" \
      $CKPT_ARG \
      --output_dir "$RUN_DIR/figures" \
      --seed       42
  done
done

# ── STEP 3: LaTeX tables ──────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: LaTeX table rows"
echo "============================================================"
python print_latex_tables.py --results_dir "$RUN_DIR" \
  | tee "$RUN_DIR/latex_tables.txt"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " All done. Run directory: $RUN_DIR"
echo ""
echo " Contents:"
find "$RUN_DIR" -type f | sort | sed 's|^|   |'
echo "============================================================"