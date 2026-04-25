"""
make_figures.py
===============
Ghép ảnh đã có trong <run_dir>/images/ thành figure tổng hợp PDF + PNG.
KHÔNG chạy lại attack — chỉ đọc ảnh đã lưu sẵn.

Cấu trúc thư mục images/ được tạo bởi evaluate_defense.py:
    images/<dataset>_<model>_<attack>/
        orig/               img_001_<attack>.png  ...
        none_None/          img_001_<attack>.png  ...
        dp_0p001/           img_001_<attack>.png  ...
        lcn_0p9/            img_001_<attack>.png  ...
        ...

Output:
    <run_dir>/figures/fig_<attack>_<dataset>_<model>.pdf
    <run_dir>/figures/fig_<attack>_<dataset>_<model>.png

Usage:
    python make_figures.py --run_dir ./results/idlg_20260420_231356
    python make_figures.py --run_dir ./results/idlg_20260420_231356 --n_show 4
"""

import os
import argparse
import glob
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Tên hiển thị cho từng condition ──────────────────────────────
# Key = tên thư mục thực tế (defense_paramstr, chữ thường)
COND_LABELS = {
    "none_none":   "No Defense",
    "none_None":   "No Defense",
    "dp_0p001":    "DP σ=1e-3",
    "dp_0p01":     "DP σ=1e-2",
    "lcn_1p1":     "LCN α=1.1",
    "lcn_0p9":     "LCN α=0.9",
    "lcn_0p7":     "LCN α=0.7",
    "lcn_0p5":     "LCN α=0.5",
}

# Thứ tự hiển thị cột — none_none LUÔN đứng đầu (cột 2, sau Original)
COND_ORDER = [
    "none_none",   # No Defense — ưu tiên cao nhất
    "none_None",   # tương thích ngược nếu thư mục cũ dùng chữ hoa
    "dp_0p001", "dp_0p01",
    "lcn_1p1", "lcn_0p9", "lcn_0p7", "lcn_0p5",
]


def find_image_sets(images_root):
    """
    Tìm tất cả bộ (dataset, model, attack) có trong images/.
    Returns list of (dataset_model_attack, path).
    """
    sets = []
    if not os.path.isdir(images_root):
        return sets
    for entry in sorted(os.listdir(images_root)):
        full = os.path.join(images_root, entry)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "orig")):
            sets.append((entry, full))
    return sets


def get_sorted_conditions(img_dir):
    """
    Trả về list các condition subdirs theo COND_ORDER.
    - none_none (No Defense) luôn đứng đầu (cột 2 sau Original)
    - Title hiển thị từ COND_LABELS, không dùng tên thư mục raw
    - Bỏ duplicate (tránh trường hợp cả none_none và none_None cùng tồn tại)
    """
    existing = set(
        d for d in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, d))
        and d not in ("orig", "comparison")
    )
    seen    = set()
    ordered = []
    for c in COND_ORDER:
        if c in existing and c not in seen:
            ordered.append(c)
            seen.add(c)
    # Thêm các cond không có trong COND_ORDER vào cuối
    for c in sorted(existing):
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def load_image(path):
    """Load PIL Image, trả về None nếu không tồn tại."""
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def find_img_files(folder, n_show):
    """Tìm tối đa n_show file PNG trong folder, sort theo tên."""
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return files[:n_show]


def make_figure(img_dir, set_name, n_show, output_dir):
    """
    Tạo figure tổng hợp cho 1 bộ (dataset_model_attack).
    Rows = ảnh, Cols = [Original | condition1 | condition2 | ...]
    """
    orig_dir = os.path.join(img_dir, "orig")
    if not os.path.isdir(orig_dir):
        print(f"  SKIP {set_name}: không có thư mục orig/")
        return

    conditions = get_sorted_conditions(img_dir)
    if not conditions:
        print(f"  SKIP {set_name}: không có condition nào")
        return

    # Tìm danh sách ảnh gốc
    orig_files = find_img_files(orig_dir, n_show)
    if not orig_files:
        print(f"  SKIP {set_name}: không có ảnh trong orig/")
        return

    n_rows = len(orig_files)
    n_cols = 1 + len(conditions)

    fig = plt.figure(figsize=(2.2 * n_cols, 2.2 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.04, wspace=0.04)

    for row, orig_path in enumerate(orig_files):
        fname = os.path.basename(orig_path)

        # Cột 0: ảnh gốc
        ax = fig.add_subplot(gs[row, 0])
        img = load_image(orig_path)
        if img:
            ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title("Original", fontsize=7, fontweight="bold", pad=3)

        # Các cột condition
        for col, cond in enumerate(conditions):
            cond_path = os.path.join(img_dir, cond, fname)
            ax = fig.add_subplot(gs[row, col + 1])
            img = load_image(cond_path)
            if img:
                ax.imshow(img)
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes,
                        color="#888888")
            ax.axis("off")
            if row == 0:
                label = COND_LABELS.get(cond, cond)
                ax.set_title(label, fontsize=6, fontweight="bold", pad=3)

    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, f"fig_{set_name}")

    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=72)
    plt.close(fig)

    print(f"  Saved: {out_base}.pdf + .png")


def main(args):
    images_root = os.path.join(args.run_dir, "images")
    output_dir  = os.path.join(args.run_dir, "figures")

    sets = find_image_sets(images_root)
    if not sets:
        print(f"Không tìm thấy ảnh trong: {images_root}")
        print("Kiểm tra lại --run_dir có đúng không.")
        return

    print(f"Found {len(sets)} image set(s):")
    for name, _ in sets:
        print(f"  {name}")
    print()

    for set_name, img_dir in sets:
        print(f"Processing: {set_name}")
        make_figure(img_dir, set_name, args.n_show, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ghép ảnh đã có thành figure tổng hợp (không chạy lại attack)")
    parser.add_argument("--run_dir", required=True,
                        help="Thư mục run đã hoàn thành "
                             "(e.g. ./results/idlg_20260420_231356)")
    parser.add_argument("--n_show", type=int, default=4,
                        help="Số ảnh tối đa mỗi row (default: 4)")
    args = parser.parse_args()
    main(args)