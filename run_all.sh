#!/usr/bin/env bash
# run_all.sh — Run the full defense capability evaluation
# Usage: bash run_all.sh [--quick]
#
# --quick  uses only 3 samples and 500 IG iterations for fast testing

set -e

QUICK=0
for arg in "$@"; do
  if [ "$arg" == "--quick" ]; then QUICK=1; fi
done

N_SAMPLES=2
IG_ITER=200
if [ $QUICK -eq 1 ]; then
  N_SAMPLES=3
  IG_ITER=500
  echo ">>> QUICK MODE: n_samples=$N_SAMPLES, ig_iter=$IG_ITER"
fi

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

echo ""
echo "============================================================"
echo " STEP 1: Quantitative evaluation (CSV)"
echo "============================================================"

#for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10"  "tinyimagenet resnet18 $CKPT_TINY"; do
for DATASET_MODEL in "tinyimagenet resnet18 $CKPT_TINY"; do
  read -r DATASET MODEL CKPT <<< "$DATASET_MODEL"
  for ATTACK in idlg; do
  #for ATTACK in ig idlg; do
    echo ""
    echo ">>> Dataset=$DATASET  Model=$MODEL  Attack=$ATTACK"
    python evaluate_defense.py \
      --dataset     "$DATASET" \
      --model       "$MODEL" \
      --attack      "$ATTACK" \
      --n_samples   "$N_SAMPLES" \
      --checkpoint  "$CKPT" \
      --output_dir  ./results \
      --seed        42
  done
done

echo ""
echo "============================================================"
echo " STEP 2: Visualization (PDF + PNG figures)"
echo "============================================================"

#for DATASET_MODEL in "cifar10 mobilenet $CKPT_CIFAR10" "tinyimagenet resnet18 $CKPT_TINY"; do
for DATASET_MODEL in  "tinyimagenet resnet18 $CKPT_TINY"; do
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
      --output_dir  ./results/figures \
      --seed        42
  done
done

echo ""
echo "============================================================"
echo " STEP 3: Print LaTeX table rows"
echo "============================================================"
#python print_latex_tables.py --results_dir ./results

echo ""
echo "All done. Results in ./results/"
