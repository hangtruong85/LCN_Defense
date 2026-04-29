#!/usr/bin/env bash
# run_idlg_parallel.sh — Chạy iDLG song song, mỗi thread xử lý 1 ảnh
#
# Dùng:
#   bash run_idlg_parallel.sh                    # full run
#   bash run_idlg_parallel.sh --quick            # 3 ảnh, iter=300
#   bash run_idlg_parallel.sh --threads 5        # giới hạn 5 threads

set -e

# ── Parse args ─────────────────────────────────────────────────
QUICK=0
N_THREADS=10
N_IMAGES=10
IDLG_ITER=500
IDLG_LR=0.1

while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)   QUICK=1;         shift ;;
    --threads) N_THREADS="$2";  shift 2 ;;
    *)         shift ;;
  esac
done

if [ $QUICK -eq 1 ]; then
  N_IMAGES=3
  IDLG_ITER=300
  N_THREADS=$N_IMAGES
  echo ">>> QUICK MODE: n_images=$N_IMAGES  iter=$IDLG_ITER  threads=$N_THREADS"
fi


CONFIGS=(
  #"mnist     lenet  none"
  "cifar10   lenet  none"
  #"cifar100  lenet  none"
  #"lfw       lenet  none"
)

#BETA_VALUES=("0.2499" "0" "-100" "1000")
BETA_VALUES=("0")

# ── Timestamped run dir ────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
#RUN_DIR="./results/idlg_${TIMESTAMP}"
RUN_DIR="./results/idlg_cifar10_beta_0"
mkdir -p "$RUN_DIR"

# ── Cố định img_seed để tất cả processes dùng cùng bộ ảnh ─────
IMG_SEED=$((RANDOM * RANDOM % 100000))

echo ""
echo "============================================================"
echo " iDLG — Parallel evaluation (${N_THREADS} threads)"
echo " Run dir  : $RUN_DIR"
echo " n_images : $N_IMAGES  (img_seed=$IMG_SEED)"
echo " iter     : $IDLG_ITER  lr=$IDLG_LR"
echo "============================================================"

cat > "$RUN_DIR/run_config.txt" << CONF
attack      : idlg
timestamp   : $TIMESTAMP
n_images    : $N_IMAGES
img_seed    : $IMG_SEED
idlg_iter   : $IDLG_ITER
idlg_lr     : $IDLG_LR
n_threads   : $N_THREADS
CONF


# ── Hàm chạy 1 job (1 ảnh, 1 config, 1 beta) ──────────────────
run_one_image() {
  local DATASET=$1
  local MODEL=$2
  local CKPT=$3
  local BETA_K=$4
  local IMG_IDX=$5

  local CKPT_ARG=""
  [ "$CKPT" != "none" ] && [ -f "$CKPT" ] && CKPT_ARG="--checkpoint $CKPT"

  local LFW_ARG=""
  [ "$DATASET" == "lfw" ] && [ -d "./data/lfw" ] && \
      LFW_ARG="--lfw_dir ./data/lfw"

  local BETA_ARG=""
  [ "$BETA_K" != "random" ] && BETA_ARG="--beta_k $BETA_K"

  local LOG_FILE="$RUN_DIR/worker_${DATASET}_${MODEL}_beta${BETA_K}_img${IMG_IDX}.log"

  python evaluate_defense_idlg.py \
    --dataset      "$DATASET" \
    --model        "$MODEL" \
    --attack       idlg \
    --n_samples    "$N_IMAGES" \
    --idlg_iter    "$IDLG_ITER" \
    --idlg_lr      "$IDLG_LR" \
    --img_seed     "$IMG_SEED" \
    --img_index    "$IMG_IDX" \
    --total_images "$N_IMAGES" \
    --run_dir      "$RUN_DIR" \
    --seed         42 \
    $CKPT_ARG \
    $LFW_ARG \
    $BETA_ARG \
    > "$LOG_FILE" 2>&1
}
export -f run_one_image
export RUN_DIR N_IMAGES IDLG_ITER IDLG_LR IMG_SEED

# ── STEP 1: Chạy song song ─────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Parallel evaluation"
echo "============================================================"

for BETA_K in "${BETA_VALUES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    read -r DATASET MODEL CKPT <<< "$CFG"
    echo ""
    echo ">>> beta_k=$BETA_K  Dataset=$DATASET  Model=$MODEL"
    echo "    Spawning ${N_IMAGES} workers (max ${N_THREADS} concurrent)..."

    for IMG_IDX in $(seq 0 $((N_IMAGES - 1))); do
      while [ "$(jobs -rp | wc -l)" -ge "$N_THREADS" ]; do
        sleep 1
      done
      run_one_image "$DATASET" "$MODEL" "$CKPT" "$BETA_K" "$IMG_IDX" &
      echo -n "    [img $((IMG_IDX+1))/$N_IMAGES pid=$!] "
    done
    echo ""

    wait
    echo "    All workers done for $DATASET/$MODEL/beta=$BETA_K"
  done
done

wait
echo ""
echo "All parallel jobs completed."

# ── STEP 2: Merge CSV ──────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Merge CSV results"
echo "============================================================"
python merge_partial_csv.py --run_dir "$RUN_DIR"

# ── STEP 3: Figures ────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 3: Summary figures"
echo "============================================================"
python make_figures.py --run_dir "$RUN_DIR" --n_show 4

echo ""
echo "============================================================"
echo " All done. Run dir: $RUN_DIR"
echo " img_seed=$IMG_SEED  (ghi lại để reproduce)"
echo " Contents:"
find "$RUN_DIR" -name "defense_capability_*.csv" | sort | sed 's|^|   |'
echo "============================================================"