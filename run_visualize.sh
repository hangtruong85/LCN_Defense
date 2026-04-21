#!/usr/bin/env bash
# run_visualize.sh — Ghép ảnh đã có thành figure tổng hợp
# KHÔNG chạy lại attack.
#
# Usage:
#   bash run_visualize.sh --dir ./results/idlg_20260420_231356
#   bash run_visualize.sh --dir ./results/idlg_20260420_231356 --n_show 6

set -e

RUN_DIR=""
N_SHOW=4

while [[ $# -gt 0 ]]; do
  case $1 in
    --dir)    RUN_DIR="$2"; shift 2 ;;
    --n_show) N_SHOW="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; shift ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  echo "ERROR: Cần truyền --dir <path>"
  echo "  Ví dụ: bash run_visualize.sh --dir ./results/idlg_20260420_231356"
  exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: Thư mục không tồn tại: $RUN_DIR"
  exit 1
fi

echo ""
echo "============================================================"
echo " Make figures from existing images"
echo " Run directory : $RUN_DIR"
echo " n_show        : $N_SHOW"
echo "============================================================"

python make_figures.py \
  --run_dir "$RUN_DIR" \
  --n_show  "$N_SHOW"

echo ""
echo "Done."