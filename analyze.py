import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SVO_LANGS = [
    "fra_Latn",  # French
    "deu_Latn",  # German
    "tha_Thai",  # Thai
    "ind_Latn",  # Indonesian
    "zho_Hans",  # Chinese (Simplified)
    "rus_Cyrl",  # Russian
    "vie_Latn",  # Vietnamese
    "por_Latn",  # Portuguese
    "swh_Latn",  # Swahili
    "fin_Latn",  # Finnish
    "swe_Latn",  # Swedish
]

SOV_LANGS = [
    "jpn_Jpan",  # Japanese
    "kor_Hang",  # Korean
    "tur_Latn",  # Turkish
    "pes_Arab",  # Persian (Farsi)
    "hin_Deva",  # Hindi
    "kat_Geor",  # Georgian
    "tam_Taml",  # Tamil
    "azj_Latn",  # Azerbaijani
    "khk_Cyrl",  # Mongolian
    "urd_Arab",  # Urdu
    "ben_Beng",  # Bengali
]


# ---------------------------------------------------------------------------
# CV helper
# ---------------------------------------------------------------------------


def cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seeds: List[int] = (42,),
) -> Tuple[float, float]:
    """Return (mean_accuracy, std_accuracy) over seeds × folds."""
    accs = []
    for seed in seeds:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y, groups=groups):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr = scaler.transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_tr, y[train_idx])
            accs.append(accuracy_score(y[val_idx], clf.predict(X_val)))
    arr = np.array(accs)
    return float(arr.mean()), float(arr.std(ddof=0))


# ---------------------------------------------------------------------------
# Column selectors
# ---------------------------------------------------------------------------


def cols_layer_head(df: pd.DataFrame, layer: int, head: int) -> List[str]:
    pat = re.compile(rf".*_l{layer:02d}_h{head:02d}(_k\d+)?$")
    return [c for c in df.columns if pat.match(c)]


def cols_layer(df: pd.DataFrame, layer: int) -> List[str]:
    pat = re.compile(rf".*_l{layer:02d}_h\d+(_k\d+)?$")
    return [c for c in df.columns if pat.match(c)]


def cols_head(df: pd.DataFrame, head: int) -> List[str]:
    pat = re.compile(rf".*_l\d+_h{head:02d}(_k\d+)?$")
    return [c for c in df.columns if pat.match(c)]


def infer_dims(df: pd.DataFrame) -> Tuple[int, int]:
    if "n_layers" in df.columns and "n_heads" in df.columns:
        return int(df["n_layers"].iloc[0]), int(df["n_heads"].iloc[0])
    layers, heads = set(), set()
    for c in df.columns:
        m = re.search(r"_l(\d+)_h(\d+)", c)
        if m:
            layers.add(int(m.group(1)))
            heads.add(int(m.group(2)))
    if not layers:
        raise ValueError("No feature columns found.")
    return max(layers) + 1, max(heads) + 1


def filter_topk(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    """Keep only eigenvalue columns with k-index < topk."""
    keep = []
    for c in df.columns:
        m = re.search(r"_k(\d+)$", c)
        if m:
            if int(m.group(1)) < topk:
                keep.append(c)
        else:
            keep.append(c)  # метаколонки не трогаем
    return df[keep]


# ---------------------------------------------------------------------------
# Standard analysis
# ---------------------------------------------------------------------------


def run_cv(
    df: pd.DataFrame, output_dir: str, n_splits: int, seeds: List[int], name: str
):
    os.makedirs(output_dir, exist_ok=True)

    y = df["y"].to_numpy()
    groups = df["sent_id"].to_numpy()
    n_layers, n_heads = infer_dims(df)
    print(f"Detected: {n_layers} layers × {n_heads} heads")
    print(f"Dataset: {len(df)} samples, class balance: {np.bincount(y)}")

    results: Dict = {}

    # 1. Full features
    print("\n[1/4] Full features")
    X_full = df[
        [c for c in df.columns if re.search(r"_l\d+_h\d+", c) or c.startswith("eig_")]
    ].to_numpy(dtype=np.float32)
    m, s = cv_accuracy(X_full, y, groups, n_splits, seeds)
    results["full"] = {"mean": round(m, 4), "std": round(s, 4)}
    print(f"  acc = {m:.4f} ± {s:.4f}")

    # 2. Per-layer
    print("\n[2/4] Per-layer")
    layer_accs = []
    for l in range(n_layers):
        cols = cols_layer(df, l)
        m, s = cv_accuracy(
            df[cols].to_numpy(dtype=np.float32), y, groups, n_splits, seeds
        )
        layer_accs.append((m, s))
        print(f"  L{l:02d}: {m:.4f} ± {s:.4f}")
    results["by_layer"] = [
        {"layer": l, "mean": round(m, 4), "std": round(s, 4)}
        for l, (m, s) in enumerate(layer_accs)
    ]

    # 3. Per-head
    print("\n[3/4] Per-head")
    head_accs = []
    for h in range(n_heads):
        cols = cols_head(df, h)
        m, s = cv_accuracy(
            df[cols].to_numpy(dtype=np.float32), y, groups, n_splits, seeds
        )
        head_accs.append((m, s))
        print(f"  H{h:02d}: {m:.4f} ± {s:.4f}")
    results["by_head"] = [
        {"head": h, "mean": round(m, 4), "std": round(s, 4)}
        for h, (m, s) in enumerate(head_accs)
    ]

    # 4. Heatmap
    print(f"\n[4/4] Heatmap ({n_layers}×{n_heads} = {n_layers * n_heads} probes)")
    heatmap = np.full((n_layers, n_heads), np.nan)
    for l in range(n_layers):
        for h in range(n_heads):
            cols = cols_layer_head(df, l, h)
            if not cols:
                continue
            m, _ = cv_accuracy(
                df[cols].to_numpy(dtype=np.float32), y, groups, n_splits, seeds
            )
            heatmap[l, h] = m
        print(f"  Layer {l:02d} done")
    results["heatmap"] = heatmap.tolist()

    with open(os.path.join(output_dir, f"{name}_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_heatmap(heatmap, output_dir, name)
    _plot_by_layer(layer_accs, output_dir, name)
    _plot_by_head(head_accs, output_dir, name)
    print(f"\nAll outputs saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_heatmap(heatmap: np.ndarray, output_dir: str, name: str):
    n_layers, n_heads = heatmap.shape
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.6), max(6, n_layers * 0.4)))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Probe Accuracy per (Layer, Head)", fontsize=13)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(range(n_heads), fontsize=8)
    ax.set_yticklabels(range(n_layers), fontsize=8)
    for i in range(n_layers):
        for j in range(n_heads):
            v = heatmap[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if v < 0.8 else "black",
                )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_heatmap.png"), dpi=150)
    plt.close()


def _plot_by_layer(layer_accs: List[Tuple[float, float]], output_dir: str, name: str):
    layers = list(range(len(layer_accs)))
    means = [m for m, _ in layer_accs]
    stds = [s for _, s in layer_accs]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, means, marker="o", linewidth=1.5, color="#2c7bb6", label="Accuracy")
    ax.fill_between(
        layers,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        color="#2c7bb6",
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Probe Accuracy by Layer", fontsize=13)
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_by_layer.png"), dpi=150)
    plt.close()


def _plot_by_head(head_accs: List[Tuple[float, float]], output_dir: str, name: str):
    heads = list(range(len(head_accs)))
    means = [m for m, _ in head_accs]
    stds = [s for _, s in head_accs]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(heads, means, marker="o", linewidth=1.5, color="#2c7bb6", label="Accuracy")
    ax.fill_between(
        heads,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        color="#2c7bb6",
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Probe Accuracy by Head", fontsize=13)
    ax.set_xticks(heads)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_by_head.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------


def leave_one_out(
    df: pd.DataFrame,
    output_dir: str,
    name: str,
    seed: int = 42,
    lang_col: str = "tgt_lang",
):
    os.makedirs(output_dir, exist_ok=True)
    name += "loo"

    feat_cols = [
        c for c in df.columns if re.search(r"_l\d+_h\d+", c) or c.startswith("eig_")
    ]
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df["y"].to_numpy()

    all_langs = SVO_LANGS + SOV_LANGS
    print(f"\nLeave-one-out: {len(all_langs)} languages, lang_col='{lang_col}'")

    records = []
    for lang in all_langs:
        test_mask = df[lang_col] == lang
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"  Skip {lang}: empty split")
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X_train), y_train)
        acc = float(accuracy_score(y_test, clf.predict(scaler.transform(X_test))))

        word_order = "SVO" if lang in SVO_LANGS else "SOV"
        print(f"  {lang.split('_')[0]:6s} ({word_order}): acc={acc:.4f}")

        records.append(
            {
                "lang": lang,
                "word_order": word_order,
                "acc": round(acc, 4),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        )

    results_df = pd.DataFrame(records)
    overall_mean = float(results_df["acc"].mean())
    overall_std = float(results_df["acc"].std(ddof=0))
    print(f"\nOverall LOO: {overall_mean:.4f} ± {overall_std:.4f}")

    results_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
        json.dump(
            {
                "overall_mean": round(overall_mean, 4),
                "overall_std": round(overall_std, 4),
                "per_lang": records,
            },
            f,
            indent=2,
        )

    print(f"Saved LOO results to {output_dir}/")
    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Run probing analysis on eigenvalue features."
    )
    p.add_argument(
        "--input", required=True, help="Path to CSV produced by build_dataset.py"
    )
    p.add_argument(
        "--output_dir", required=True, help="Directory for results and plots"
    )
    p.add_argument(
        "--name", required=True, help="Prefix for output files (e.g. mode1_600M)"
    )
    p.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2],
        help="1 = eng→X (lang_col=tgt_lang), 2 = X→eng (lang_col=src_lang)",
    )
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument(
        "--analysis",
        nargs="+",
        choices=["cv", "loo", "all"],
        default=["all"],
        help="Which analyses to run",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Use only top-k eigenvalues (0..topk-1). Ignored for baseline features.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    lang_col = "tgt_lang" if args.mode == 1 else "src_lang"
    run_all = "all" in args.analysis

    if run_all or "cv" in args.analysis:
        if args.topk < 12:
            df = filter_topk(df, args.topk)
            print(f"Using top-{args.topk} eigenvalues per (layer, head)")
        run_cv(df, args.output_dir, args.n_splits, args.seeds, args.name)

    if run_all or "loo" in args.analysis:
        leave_one_out(
            df, args.output_dir, args.name, seed=args.seeds[0], lang_col=lang_col
        )
