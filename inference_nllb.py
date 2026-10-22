import argparse
import os
import tarfile
import urllib.request
from typing import List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ENG_CODE = "eng_Latn"

AVAILABLE_MODELS = {
    "600M": "facebook/nllb-200-distilled-600M",
    "1.3B": "facebook/nllb-200-distilled-1.3B",
    "3.3B": "facebook/nllb-200-3.3B",
}

OTHER_LANGS: List[str] = [
    # SVO
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
    # SOV
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

LANG_CODES = [ENG_CODE] + OTHER_LANGS


def load_or_download_flores_dev(
    root_path: str,
    url: str = "https://tinyurl.com/flores200dataset",
) -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    dataset_dir = os.path.join(root_path, "flores200_dataset")
    archive_path = os.path.join(root_path, "flores200_dataset.tar.gz")

    if not os.path.exists(dataset_dir):
        print("Dataset not found. Downloading...")
        urllib.request.urlretrieve(url, archive_path)
        print("Download complete:", archive_path)
        print("Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root_path)
        print("Extraction completed.")
        os.remove(archive_path)

    dev_path = os.path.join(dataset_dir, "dev")
    rows = []
    for lang in LANG_CODES:
        file_path = os.path.join(dev_path, f"{lang}.dev")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rows.append({"id": i, "lang": lang, "text": line})

    return pd.DataFrame(rows)


def dump_attentions_all_pairs(
    df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    num_beams: int = 4,
    max_length: int = 256,
    limit_sentences: int = None,
):
    model_slug = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_slug)
    attn_root = os.path.join(output_dir, "attentions")
    trans_root = os.path.join(output_dir, "translations")

    os.makedirs(attn_root, exist_ok=True)
    os.makedirs(trans_root, exist_ok=True)

    print("Loading model and tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=LANG_CODES[0])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for src_lang in LANG_CODES:
        src_df = df[df["lang"] == src_lang].sort_values("id")
        if limit_sentences:
            src_df = src_df.head(limit_sentences)

        texts = src_df["text"].tolist()
        ids = src_df["id"].tolist()
        n_sent = len(ids)

        for tgt_lang in LANG_CODES:
            is_eng_to_other = src_lang == ENG_CODE and tgt_lang != ENG_CODE
            is_other_to_eng = src_lang != ENG_CODE and tgt_lang == ENG_CODE
            if not (is_eng_to_other or is_other_to_eng):
                continue

            attn_pair_dir = os.path.join(attn_root, f"{src_lang}-{tgt_lang}")
            trans_pair_dir = os.path.join(trans_root, f"{src_lang}-{tgt_lang}")
            os.makedirs(attn_pair_dir, exist_ok=True)
            os.makedirs(trans_pair_dir, exist_ok=True)

            tokenizer.src_lang = src_lang
            tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

            print(f"Translating & dumping {src_lang}->{tgt_lang}")

            for idx in tqdm(
                range(n_sent),
                desc=f"{src_lang}->{tgt_lang}",
                dynamic_ncols=True,
                leave=False,
            ):
                sent_id = ids[idx]
                text = texts[idx]

                attn_path = os.path.join(
                    attn_pair_dir, f"{src_lang}-{tgt_lang}-{sent_id}.pt"
                )
                trans_path = os.path.join(
                    trans_pair_dir, f"{src_lang}-{tgt_lang}-{sent_id}.txt"
                )

                if os.path.exists(attn_path) and os.path.exists(trans_path):
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)

                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        forced_bos_token_id=tgt_lang_id,
                        output_attentions=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                translation_text = tokenizer.decode(
                    gen_out.sequences[0],
                    skip_special_tokens=True,
                )
                with open(trans_path, "w", encoding="utf-8") as f:
                    f.write(translation_text)

                beam_indices = gen_out.beam_indices
                cross_att = gen_out.cross_attentions
                num_steps_gen = len(cross_att)
                num_layers = len(cross_att[0])
                steps_to_use = min(num_steps_gen, beam_indices.shape[1])

                if steps_to_use == 0:
                    del gen_out, beam_indices, cross_att
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                best_beam_path = beam_indices[0, :steps_to_use].tolist()

                per_layer_list = []
                for layer_idx in range(num_layers):
                    per_step = []
                    for t in range(steps_to_use):
                        beam_idx_t = best_beam_path[t]
                        att_t_l = cross_att[t][layer_idx][beam_idx_t, :, 0, :]
                        per_step.append(att_t_l.detach().to("cpu", dtype=torch.float16))
                    layer_att = torch.stack(per_step, dim=0).permute(1, 0, 2)
                    per_layer_list.append(layer_att)

                cross_attn_all = torch.stack(per_layer_list, dim=0)
                torch.save(cross_attn_all, attn_path)

                del gen_out, beam_indices, cross_att, cross_attn_all
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            print(f"  -> Done {src_lang} -> {tgt_lang}")

    print("\nAll translations and attentions dumped.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump NLLB cross-attention maps and translations for FLORES-200 dev set."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="600M",
        choices=list(AVAILABLE_MODELS.keys()),
        help=f"Model size. One of: {list(AVAILABLE_MODELS.keys())}. Default: 600M",
    )
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit sentences per language for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = AVAILABLE_MODELS[args.model]
    print(f"Model: {args.model} ({model_name})")

    print("Loading FLORES dev set...")
    df = load_or_download_flores_dev(args.root_dir)
    print(f"Loaded {len(df)} rows.")

    dump_attentions_all_pairs(
        df=df,
        output_dir=args.out_dir,
        model_name=model_name,
        num_beams=args.num_beams,
        max_length=args.max_length,
        limit_sentences=args.limit,
    )


if __name__ == "__main__":
    main()
