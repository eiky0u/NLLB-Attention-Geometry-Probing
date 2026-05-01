# Cross-Attention Eigenvalues Know the Word Order in Multilingual Machine Translation

Code for the paper *"Cross-Attention Eigenvalues Know the Word Order in Multilingual Machine Translation"*.

We probe the cross-attention mechanism of [NLLB-200](https://arxiv.org/abs/2207.04672) to investigate whether its geometry encodes target-language word order (SVO vs. SOV). Features are eigenvalues of per-layer-head Gram matrices; classification is done with a linear logistic regression probe.

---

## Requirements

```bash
pip install -r requirements.txt
```

Models are downloaded automatically from Hugging Face on first run. The FLORES-200 dataset is downloaded automatically as well.

---

## Pipeline

The pipeline consists of three scripts that must be run in order.

### Step 1 — Dump cross-attention maps

Translates all sentences in the FLORES-200 dev set and saves cross-attention tensors to disk.

```bash
python inference_nllb.py \
    --model 600M \
    --root_dir ./data \
    --out_dir ./outputs
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `600M` | Model size: `600M`, `1.3B`, or `3.3B` |
| `--root_dir` | `./data` | Directory for the FLORES-200 dataset |
| `--out_dir` | `./outputs` | Directory for attention tensors and translations |
| `--num_beams` | `4` | Beam search width |
| `--max_length` | `256` | Maximum generation length |
| `--limit` | `None` | Limit sentences per language (useful for testing) |

Output structure:
```
outputs/
  nllb-200-distilled-600M/
    attentions/
      eng_Latn-fra_Latn/
        eng_Latn-fra_Latn-0.pt
        ...
    translations/
      eng_Latn-fra_Latn/
        ...
```

Each `.pt` file is a tensor of shape `[L, H, T, S]` (layers × heads × target tokens × source tokens).

---

### Step 2 — Extract eigenvalue features

Computes the top-$k$ eigenvalues of the Gram matrix $AA^\top$ for each layer-head pair and saves a feature matrix as CSV.

```bash
python build_dataset.py \
    --attn_root ./outputs/nllb-200-distilled-600M/attentions \
    --output ./outputs/mode1_600M.csv \
    --mode 1
```

| Argument | Default | Description |
|---|---|---|
| `--attn_root` | required | Path to the `attentions/` directory from Step 1 |
| `--output` | required | Output CSV path |
| `--mode` | `1` | `1` = eng→X (target varies), `2` = X→eng (source varies) |
| `--topk` | `12` | Number of eigenvalues per layer-head pair |
| `--limit` | `None` | Limit sentences per language pair |
| `--overwrite` | flag | Overwrite existing CSV |

Run once per model size and per mode (e.g. `mode1_600M.csv`, `mode2_600M.csv`).

---

### Step 3 — Run probing analysis

Trains a logistic regression probe and evaluates it with cross-validation and leave-one-language-out (LOO).

```bash
python analyze.py \
    --input ./outputs/mode1_600M.csv \
    --output_dir ./outputs/results \
    --name mode1_600M \
    --mode 1
```

| Argument | Default | Description |
|---|---|---|
| `--input` | required | CSV from Step 2 |
| `--output_dir` | required | Directory for results and plots |
| `--name` | required | Prefix for output files |
| `--mode` | `1` | `1` = eng→X, `2` = X→eng |
| `--n_splits` | `5` | Number of CV folds |
| `--seeds` | `42` | One or more random seeds (space-separated) |
| `--analysis` | `all` | Analyses to run: `cv`, `loo`, or `all` |

Output files:
```
results/
  mode1_600M_summary.json   # CV accuracy: full, per-layer, per-head, heatmap
  mode1_600M_heatmap.png
  mode1_600M_by_layer.png
  mode1_600M_by_head.png
  mode1_600M_loo.csv            # LOO per-language accuracy
  mode1_600M_loo.json           # LOO summary
```

---
