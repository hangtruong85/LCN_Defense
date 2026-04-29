"""
merge_partial_csv.py
====================
Gộp các file CSV partial (từ chạy song song) thành CSV tổng hợp.

Cấu trúc CSV partial (từ evaluate_defense_idlg.py):
    attack,defense,param,dataset,model,PSNR_mean,PSNR_std,SSIM_mean,SSIM_std,LPIPS_mean,LPIPS_std

Dùng:
    python merge_partial_csv.py --run_dir ./results/idlg_beta_20260425_230411
    python merge_partial_csv.py --run_dir ./results/idlg_beta_20260425_230411 --min_images 5
"""

import os, glob, csv, argparse
import numpy as np
from collections import defaultdict


def merge(run_dir, min_images=1):
    pattern   = os.path.join(run_dir, "partial_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No partial CSV files found in: {run_dir}")
        return

    print(f"Found {len(csv_files)} partial CSV files in: {run_dir}")

    # Group by key = tên file bỏ phần _img<NNN>
    groups = defaultdict(list)
    for f in csv_files:
        basename = os.path.basename(f)
        stem  = basename.replace("partial_", "").replace(".csv", "")
        parts = stem.split("_")
        key   = "_".join(parts[:-1])
        groups[key].append(f)

    n_merged = 0
    for key, files in sorted(groups.items()):
        n_files = len(files)
        print(f"\n[{key}] — {n_files} partial file(s)")

        if n_files < min_images:
            print(f"  SKIP: {n_files} < {min_images} images")
            continue

        # Gom metrics theo (defense, param)
        all_rows = defaultdict(lambda: defaultdict(list))
        for f in sorted(files):
            with open(f, newline="") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    cond = (row["defense"], row["param"])
                    for metric in ["PSNR", "SSIM", "LPIPS"]:
                        val = row.get(f"{metric}_mean", "")
                        try:
                            if val and val not in ("-", ""):
                                all_rows[cond][metric].append(float(val))
                        except ValueError:
                            pass

        # Parse key → dataset, model, beta_k
        parts     = key.split("_")
        beta_part = [p for p in parts if p.startswith("beta")]
        beta_k    = beta_part[0].replace("beta", "") if beta_part else "?"
        try:
            beta_idx = next(i for i, p in enumerate(parts) if p.startswith("beta"))
            model    = parts[beta_idx - 1]
            dataset  = "_".join(parts[:beta_idx - 1])
        except StopIteration:
            model   = parts[-1]
            dataset = "_".join(parts[:-1])

        # Ghi CSV tổng hợp
        out_csv = os.path.join(run_dir,
            f"defense_capability_{dataset}_{model}_beta{beta_k}.csv")
        with open(out_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "defense", "param", "beta_k",
                "PSNR_mean", "PSNR_std",
                "SSIM_mean", "SSIM_std",
                "LPIPS_mean", "LPIPS_std",
                "n_images",
            ])
            for (defense, param), metrics in all_rows.items():
                n = len(metrics.get("PSNR", []))
                def _s(m):
                    v = metrics.get(m, [])
                    return (f"{np.mean(v):.4f}", f"{np.std(v):.4f}") if v else ("-", "-")
                pm, ps = _s("PSNR");  sm, ss = _s("SSIM");  lm, ls = _s("LPIPS")
                writer.writerow([defense, param, beta_k,
                                  pm, ps, sm, ss, lm, ls, n])

        print(f"  Saved: {out_csv}")
        print(f"  {'defense':<8} {'param':<8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'n':>4}")
        print(f"  {'-'*46}")
        for (defense, param), metrics in sorted(all_rows.items()):
            p = np.mean(metrics.get("PSNR",  [0]))
            s = np.mean(metrics.get("SSIM",  [0]))
            l = np.mean(metrics.get("LPIPS", [0]))
            n = len(metrics.get("PSNR", []))
            print(f"  {defense:<8} {str(param):<8} {p:>8.3f} {s:>8.4f} {l:>8.4f} {n:>4}")
        n_merged += 1

    print(f"\nDone. Merged {n_merged} group(s).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",    required=True)
    p.add_argument("--min_images", type=int, default=1)
    args = p.parse_args()
    if not os.path.isdir(args.run_dir):
        print(f"ERROR: {args.run_dir} not found"); raise SystemExit(1)
    merge(args.run_dir, min_images=args.min_images)