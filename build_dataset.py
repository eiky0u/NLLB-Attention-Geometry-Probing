import argparse
import os
import re
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORD_ORDER_MAP: Dict[str, str] = {
    "eng_Latn": "SVO",
    "fra_Latn": "SVO",
    "deu_Latn": "SVO",
    "tha_Thai": "SVO",
    "ind_Latn": "SVO",
    "zho_Hans": "SVO",
    "rus_Cyrl": "SVO",
    "vie_Latn": "SVO",
    "por_Latn": "SVO",
    "swh_Latn": "SVO",
    "fin_Latn": "SVO",
    "swe_Latn": "SVO",
    "jpn_Jpan": "SOV",
    "kor_Hang": "SOV",
    "tur_Latn": "SOV",
    "pes_Arab": "SOV",
    "hin_Deva": "SOV",
    "kat_Geor": "SOV",
    "tam_Taml": "SOV",
    "azj_Latn": "SOV",
    "khk_Cyrl": "SOV",
    "urd_Arab": "SOV",
    "ben_Beng": "SOV",
}

PAIR_DIR_RE = re.compile(r"^([a-z]{3}_[A-Za-z]+)-([a-z]{3}_[A-Za-z]+)$")

# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_attention_files(
    attn_root: str, mode: int, limit_per_pair: Optional[int] = None
):
    out = []

    for pair in sorted(os.listdir(attn_root)):
        pair_path = os.path.join(attn_root, pair)
        if not os.path.isdir(pair_path):
            continue

        m = PAIR_DIR_RE.match(pair)
        if not m:
            continue

        src, tgt = m.group(1), m.group(2)

        if mode == 1 and (src != "eng_Latn" or tgt == "eng_Latn"):
            continue
        if mode == 2 and (tgt != "eng_Latn" or src == "eng_Latn"):
            continue

        files = sorted(f for f in os.listdir(pair_path) if f.endswith(".pt"))
        if limit_per_pair:
            files = files[:limit_per_pair]

        for fn in files:
            sid = re.search(r"(\d+)\.pt$", fn)
            if sid:
                out.append((src, tgt, int(sid.group(1)), os.path.join(pair_path, fn)))

    return out


# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------


def compute_eigenvalues(attn, topk: int, eps=1e-8):
    A = attn.float()
    A = A / (A.sum(dim=-1, keepdim=True) + eps)

    M = A @ A.transpose(-1, -2)
    evals = torch.linalg.eigvalsh(M).clamp_min(0.0)
    evals = evals / (evals.sum(dim=-1, keepdim=True) + eps)
    evals = torch.sort(evals, descending=True).values

    if evals.size(-1) < topk:
        pad = torch.zeros((*evals.shape[:-1], topk - evals.size(-1)), dtype=evals.dtype)
        evals = torch.cat([evals, pad], dim=-1)

    return evals[..., :topk].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline features (atomic)
# ---------------------------------------------------------------------------


def prepare_attention(A):
    return A / (A.sum(dim=-1, keepdim=True) + 1e-8).detach().cpu().numpy().astype(
        np.float32
    )


def entropy(A):
    logA = torch.log(A + 1e-8)
    return (
        -(A * logA).sum(dim=-1).mean(dim=-1).detach().cpu().numpy().astype(np.float32)
    )


def confidence(A):
    return A.max(dim=-1).values.mean(dim=-1).detach().cpu().numpy().astype(np.float32)


def variance(A):
    S = A.size(-1)
    j = torch.arange(S, device=A.device, dtype=A.dtype)
    mu = (A * j).sum(dim=-1, keepdim=True)
    return (
        -(A * (mu - j) ** 2).sum(dim=(-2, -1)).detach().cpu().numpy().astype(np.float32)
    )


def coverage(A):
    col = A.sum(dim=-2)
    return (col**2).sum(dim=-1)


FEATURE_REGISTRY: Dict[str, Callable] = {
    "entropy": entropy,
    "confidence": confidence,
    "variance": variance,
    "coverage": coverage,
}

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


def eig_cols(L, H, K):
    return [
        f"eig_l{l:02d}_h{h:02d}_k{k:02d}"
        for l in range(L)
        for h in range(H)
        for k in range(K)
    ]


def feat_cols(L, H, feats):
    cols = []
    for l in range(L):
        for h in range(H):
            for f in feats:
                cols.append(f"{f}_l{l:02d}_h{h:02d}")
    return cols


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def label(src, tgt):
    return int(WORD_ORDER_MAP[src] == WORD_ORDER_MAP[tgt])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(
    attn_root, output_csv, mode, features, overwrite=False, topk=12, limit=None
):

    if os.path.exists(output_csv) and not overwrite:
        print("Exists, use --overwrite")
        return

    files = collect_attention_files(attn_root, mode, limit)
    print(f"Files: {len(files)}")

    X, metas = [], []
    L = H = None

    for src, tgt, sid, fp in tqdm(files):

        attn = torch.load(fp, map_location="cpu")

        if L is None:
            L, H = attn.shape[:2]

        A = prepare_attention(attn)

        parts = []

        with torch.inference_mode():

            if "eigenvalues" in features:
                parts.append(compute_eigenvalues(attn, topk).flatten())

            for f in features:
                if f in FEATURE_REGISTRY:
                    parts.append(FEATURE_REGISTRY[f](A).flatten())

        X.append(np.concatenate(parts))

        metas.append(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "pair": f"{src}-{tgt}",
                "sent_id": sid,
                "y": label(src, tgt),
                "n_layers": L,
                "n_heads": H,
            }
        )

    X = np.stack(X)

    cols = []

    if "eigenvalues" in features:
        cols += eig_cols(L, H, topk)

    for f in features:
        if f in FEATURE_REGISTRY:
            cols += feat_cols(L, H, [f])

    assert X.shape[1] == len(cols)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    df = pd.concat([pd.DataFrame(metas), pd.DataFrame(X, columns=cols)], axis=1)
    df.to_csv(output_csv, index=False)

    print(f"Saved → {output_csv}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--attn_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", type=int, choices=[1, 2], default=1)

    p.add_argument(
        "--features",
        nargs="+",
        default=["eigenvalues"],
        choices=["eigenvalues", "entropy", "confidence", "variance", "coverage"],
    )

    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_dataset(
        attn_root=args.attn_root,
        output_csv=args.output,
        mode=args.mode,
        features=args.features,
        topk=args.topk,
        limit=args.limit,
        overwrite=args.overwrite,
    )
