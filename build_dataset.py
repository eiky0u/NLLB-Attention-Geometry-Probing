import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORD_ORDER_MAP: Dict[str, str] = {
    "eng_Latn": "SVO",
    # SVO
    "fra_Latn": "SVO",  # French
    "deu_Latn": "SVO",  # German
    "tha_Thai": "SVO",  # Thai
    "ind_Latn": "SVO",  # Indonesian
    "zho_Hans": "SVO",  # Chinese (Simplified)
    "rus_Cyrl": "SVO",  # Russian
    "vie_Latn": "SVO",  # Vietnamese
    "por_Latn": "SVO",  # Portuguese
    "swh_Latn": "SVO",  # Swahili
    "fin_Latn": "SVO",  # Finnish
    "swe_Latn": "SVO",  # Swedish
    # SOV
    "jpn_Jpan": "SOV",  # Japanese
    "kor_Hang": "SOV",  # Korean
    "tur_Latn": "SOV",  # Turkish
    "pes_Arab": "SOV",  # Persian (Farsi)
    "hin_Deva": "SOV",  # Hindi
    "kat_Geor": "SOV",  # Georgian
    "tam_Taml": "SOV",  # Tamil
    "azj_Latn": "SOV",  # Azerbaijani
    "khk_Cyrl": "SOV",  # Mongolian
    "urd_Arab": "SOV",  # Urdu
    "ben_Beng": "SOV",  # Bengali
}

PAIR_DIR_RE = re.compile(r"^([a-z]{3}_[A-Za-z]+)-([a-z]{3}_[A-Za-z]+)$")


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_attention_files(
    attn_root: str,
    mode: int,
    limit_per_pair: Optional[int] = None,
) -> List[Tuple[str, str, int, str]]:
    out: List[Tuple[str, str, int, str]] = []

    for pair in sorted(d for d in os.listdir(attn_root)
                       if os.path.isdir(os.path.join(attn_root, d))):
        m = PAIR_DIR_RE.match(pair)
        if not m:
            continue
        src, tgt = m.group(1), m.group(2)

        if mode == 1:
            if src != "eng_Latn" or tgt == "eng_Latn":
                continue
        elif mode == 2:
            if tgt != "eng_Latn" or src == "eng_Latn":
                continue
        else:
            raise ValueError(f"Unknown mode {mode}. Use 1 or 2.")

        pair_path = os.path.join(attn_root, pair)
        files = sorted(f for f in os.listdir(pair_path) if f.endswith(".pt"))
        if limit_per_pair is not None:
            files = files[:int(limit_per_pair)]

        for fn in files:
            sid = re.search(r"(\d+)\.pt$", fn)
            if sid:
                out.append((src, tgt, int(sid.group(1)),
                            os.path.join(pair_path, fn)))

    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_eigenvalues(
    attn: torch.Tensor,
    topk: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute top-k eigenvalues per (layer, head).

    Args:
        attn: [L, H, T, S]
    Returns:
        ndarray of shape [L, H, topk]
    """
    A = attn.float()
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    M = A @ A.transpose(-1, -2)                          # [L, H, T, T]
    evals = torch.linalg.eigvalsh(M).clamp_min(0.0)      # [L, H, T]
    evals = evals / (evals.sum(dim=-1, keepdim=True) + eps)
    evals = torch.sort(evals, descending=True, dim=-1).values

    if evals.size(-1) >= topk:
        top = evals[..., :topk]
    else:
        pad = torch.zeros((*evals.shape[:-1], topk - evals.size(-1)), dtype=evals.dtype)
        top = torch.cat([evals, pad], dim=-1)

    return top.detach().cpu().numpy().astype(np.float32)  # [L, H, topk]


def make_column_names(n_layers: int, n_heads: int, topk: int) -> List[str]:
    return [
        f"eig_l{l:02d}_h{h:02d}_k{k:02d}"
        for l in range(n_layers)
        for h in range(n_heads)
        for k in range(topk)
    ]


def label(src: str, tgt: str) -> int:
    return 1 if WORD_ORDER_MAP[src] == WORD_ORDER_MAP[tgt] else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    attn_root: str,
    output_csv: str,
    mode: int,
    topk: int,
    limit_per_pair: Optional[int],
    overwrite: bool,
):
    if os.path.exists(output_csv) and not overwrite:
        print(f"Dataset already exists at {output_csv}. Use --overwrite to rebuild.")
        return

    files = collect_attention_files(attn_root, mode, limit_per_pair)
    if not files:
        raise RuntimeError(f"No .pt files found under {attn_root} for mode {mode}.")
    print(f"Found {len(files)} attention files across "
          f"{len(set(f'{s}-{t}' for s,t,_,_ in files))} pairs.")

    metas, X_list = [], []
    n_layers = n_heads = None

    for src, tgt, sent_id, fp in tqdm(files, desc="Extracting eigenvalues"):
        attn = torch.load(fp, map_location="cpu")  # [L, H, T, S]

        if n_layers is None:
            n_layers, n_heads = attn.shape[0], attn.shape[1]
            print(f"Detected: {n_layers} layers, {n_heads} heads, topk={topk}")

        with torch.inference_mode():
            feats = compute_eigenvalues(attn, topk=topk)  # [L, H, topk]

        X_list.append(feats.flatten())
        metas.append({
            "src_lang": src,
            "tgt_lang": tgt,
            "pair":     f"{src}-{tgt}",
            "sent_id":  sent_id,
            "y":        label(src, tgt),
            "n_layers": n_layers,
            "n_heads":  n_heads,
        })

    X = np.stack(X_list)
    feat_cols = make_column_names(n_layers, n_heads, topk)
    assert X.shape[1] == len(feat_cols), f"Shape mismatch: {X.shape[1]} != {len(feat_cols)}"

    df = pd.concat([pd.DataFrame(metas), pd.DataFrame(X, columns=feat_cols)], axis=1)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows → {output_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_root", required=True, help="Path to attentions/ directory")
    p.add_argument("--output",    required=True, help="Output CSV path")
    p.add_argument("--mode",      type=int, default=1, choices=[1, 2])
    p.add_argument("--topk",      type=int, default=12)
    p.add_argument("--limit",     type=int, default=None, help="Limit sentences per pair")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        attn_root=args.attn_root,
        output_csv=args.output,
        mode=args.mode,
        topk=args.topk,
        limit_per_pair=args.limit,
        overwrite=args.overwrite,
    )
