#!/usr/bin/env bash
# run_ig_inversefed.sh — IG defense evaluation dùng inversefed (code gốc tác giả)
#
# Dùng:
#   bash run_ig_inversefed.sh              # full run
#   bash run_ig_inversefed.sh --quick      # test nhanh (2 ảnh, 500 iter, 1 restart)

set -e

QUICK=""
for arg in "$@"; do
  [ "$arg" == "--quick" ] && QUICK="--quick"
done

CKPT_CIFAR10="./checkpoints/cifar10_mobilenet.pth"
CKPT_TINY="./checkpoints/tinyimagenet_resnet18.pth"

# Configs: dataset  model  checkpoint
CONFIGS=(
  #"cifar10      ResNet18  none"
  #"cifar10      MobileNet $CKPT_CIFAR10"
  "tinyimagenet      MobileNet $CKPT_CIFAR10"
  #"tinyimagenet ResNet18  $CKPT_TINY"
)

echo ""
echo "============================================================"
echo " IG Attack — inversefed (Geiping et al. 2020)"
echo " cost_fn=sim  signed=True  boxed=True"
echo " iter=24000   restarts=8   lr=0.1  TV=1e-4"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
  read -r DATASET MODEL CKPT <<< "$CFG"
  CKPT_ARG=""
  [ "$CKPT" != "none" ] && [ -f "$CKPT" ] && CKPT_ARG="--checkpoint $CKPT"

  echo ""
  echo ">>> Dataset=$DATASET  Model=$MODEL"
  python run_ig_inversefed.py \
    --dataset    "$DATASET" \
    --model      "$MODEL" \
    --n_samples  10 \
    --restarts   8 \
    --ig_iter    24000 \
    --cost_fn    sim \
    --tv         1e-4 \
    $CKPT_ARG \
    --output_dir ./results \
    $QUICK
done

echo ""
echo "Done."