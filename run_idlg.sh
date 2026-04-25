#!/usr/bin/env bash
# run_idlg_beta.sh — Chạy iDLG với nhiều giá trị beta_k khác nhau
# Thử nghiệm beta_k: dương, bằng 0, âm

set -e

QUICK=0
for arg in "$@"; do
  [ "$arg" == "--quick" ] && QUICK=1
done

N_SAMPLES=10
IDLG_ITER=500
IDLG_LR=0.1
[ $QUICK -eq 1 ] && N_SAMPLES=3 && IDLG_ITER=300

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="./results/idlg_beta_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo " iDLG — Beta sensitivity evaluation"
echo " Run dir: $RUN_DIR"
echo "============================================================"

# Các dataset/model configs
CONFIGS=(
  "mnist    lenet  none"
  "cifar10 lenet  none"
  "cifar100 lenet  none"
  "lfw      lenet  none"
)

# Các giá trị beta_k cần thử: >0, =0, <0
BETA_VALUES=("0.2499" "0" "-100" "1000")

for BETA_K in "${BETA_VALUES[@]}"; do
  echo ""
  echo ">>> beta_k = $BETA_K"
  for CFG in "${CONFIGS[@]}"; do
    read -r DATASET MODEL CKPT <<< "$CFG"
    CKPT_ARG=""
    [ "$CKPT" != "none" ] && [ -f "$CKPT" ] && CKPT_ARG="--checkpoint $CKPT"

    LFW_ARG=""
    [ "$DATASET" == "lfw" ] && [ -d "./data/lfw" ] && \
        LFW_ARG="--lfw_dir ./data/lfw"

    echo "  Dataset=$DATASET  Model=$MODEL"
    python evaluate_defense_idlg.py \
      --dataset    "$DATASET" \
      --model      "$MODEL" \
      --attack     idlg \
      --n_samples  "$N_SAMPLES" \
      --idlg_iter  "$IDLG_ITER" \
      --idlg_lr    "$IDLG_LR" \
      --beta_k     "$BETA_K" \
      $CKPT_ARG \
      $LFW_ARG \
      --run_dir    "$RUN_DIR" \
      --seed       42
  done
done

echo ""
echo "============================================================"
echo " STEP 2: Make figures"
echo "============================================================"
python make_figures.py --run_dir "$RUN_DIR" --n_show 4

echo ""
echo "All done. Run dir: $RUN_DIR"