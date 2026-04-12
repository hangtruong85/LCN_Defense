#!/usr/bin/env bash
# train_all.sh — Train FL models and save checkpoints
# Usage:
#   bash train_all.sh              # full training (100 rounds)
#   bash train_all.sh --quick      # quick test (5 rounds, 5 clients)
#   bash train_all.sh --cifar10    # only CIFAR-10
#   bash train_all.sh --tiny       # only Tiny-ImageNet

set -e

QUICK=0
DO_CIFAR=1
DO_TINY=1

for arg in "$@"; do
  case $arg in
    --quick)   QUICK=1    ;;
    --cifar10) DO_TINY=0  ;;
    --tiny)    DO_CIFAR=0 ;;
  esac
done

ROUNDS=100
N_CLIENTS=5
if [ $QUICK -eq 1 ]; then
  ROUNDS=5
  N_CLIENTS=5
  echo ">>> QUICK MODE: rounds=$ROUNDS"
fi

CKPT_DIR="./checkpoints"
LOG_DIR="./results"

echo ""
echo "============================================================"
echo " FL Training — generating checkpoints for defense evaluation"
echo " Checkpoints will be saved to: $CKPT_DIR"
echo "============================================================"

# ── CIFAR-10 / MobileNet ─────────────────────────────────────────
if [ $DO_CIFAR -eq 1 ]; then
  echo ""
  echo ">>> [1/2] CIFAR-10 / MobileNet  (${N_CLIENTS} clients, ${ROUNDS} rounds)"
  python train_fl.py \
    --dataset      cifar10 \
    --model        mobilenet \
    --defense      none \
    --n_clients    $N_CLIENTS \
    --rounds       $ROUNDS \
    --local_epochs 1 \
    --local_bs     64 \
    --lr           1e-3 \
    --ckpt_dir     $CKPT_DIR \
    --output_dir   $LOG_DIR \
    --seed         42

  echo ""
  echo "    Checkpoint saved: $CKPT_DIR/cifar10_mobilenet.pth"
fi

# ── Tiny-ImageNet / ResNet18 ─────────────────────────────────────
if [ $DO_TINY -eq 1 ]; then
  echo ""
  echo ">>> [2/2] Tiny-ImageNet / ResNet18  (${N_CLIENTS} clients, ${ROUNDS} rounds)"
  python train_fl.py \
    --dataset      tinyimagenet \
    --model        resnet18 \
    --defense      none \
    --n_clients    $N_CLIENTS \
    --rounds       $ROUNDS \
    --local_epochs 1 \
    --local_bs     64 \
    --lr           1e-3 \
    --ckpt_dir     $CKPT_DIR \
    --output_dir   $LOG_DIR \
    --seed         42

  echo ""
  echo "    Checkpoint saved: $CKPT_DIR/tinyimagenet_resnet18.pth"
fi

echo ""
echo "============================================================"
echo " All training done."
echo " Checkpoints:"
ls -lh $CKPT_DIR/*.pth 2>/dev/null || echo "  (none found)"
echo ""
echo " Next step — run defense evaluation:"
echo "   bash run_all.sh"
echo "============================================================"