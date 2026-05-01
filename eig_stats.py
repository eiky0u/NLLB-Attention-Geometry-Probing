import argparse
import re
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with eigenvalue features")
    p.add_argument("--zero_thresh", type=float, default=1e-6,
                   help="Threshold below which eigenvalue is considered zero")
    args = p.parse_args()

    df = pd.read_csv(args.input)

    # Find all eigenvalue columns and extract k index
    eig_cols = [c for c in df.columns if re.match(r"eig_l\d+_h\d+_k\d+$", c)]
    if not eig_cols:
        raise ValueError("No eigenvalue columns found (expected format: eig_lXX_hXX_kXX)")

    k_max = max(int(re.search(r"_k(\d+)$", c).group(1)) for c in eig_cols) + 1
    print(f"Found {len(eig_cols)} eigenvalue columns, k = 0..{k_max - 1}\n")

    rows = []
    for k in range(k_max):
        # All columns for this k index (across all layers and heads)
        cols_k = [c for c in eig_cols if re.search(r"_k(\d+)$", c).group(1) == f"{k:02d}"]
        vals = df[cols_k].values.flatten()

        mean_val = float(np.mean(vals))
        n_zero = int(np.sum(vals < args.zero_thresh))
        total = len(vals)
        pct_zero = 100.0 * n_zero / total

        rows.append({
            "k": k,
            "mean": round(mean_val, 6),
            "n_zero": n_zero,
            "total": total,
            "pct_zero": round(pct_zero, 2),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    out_path = args.input.replace(".csv", "_eig_stats.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
