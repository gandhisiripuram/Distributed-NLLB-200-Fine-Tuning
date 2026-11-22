"""
Score an English to Telugu model on FLORES+ with BLEU, chrF and chrF++.

Run ONCE per model on devtest. Use dev only for choosing epochs (train_ddp.py
already does that). Running devtest many times and keeping the best result
would turn it into a tuning set, and the score would no longer be fair.

    python src/evaluate.py --model_path facebook/nllb-200-distilled-600M --tag base
    python src/evaluate.py --model_path ./nllb-telugu-finetuned --tag finetuned

Significance test (paired bootstrap), after both runs:
    sacrebleu results/refs_devtest.txt -i results/base_devtest.txt results/finetuned_devtest.txt \
        -m bleu chrf --chrf-word-order 2 --paired-bs

The bootstrap test resamples the test sentences many times and checks how
often the fine-tuned model still beats the baseline. A small gain on 1,012
sentences can be luck; the p-value tells you whether it probably is not.

Requires a Hugging Face token with access to openlanguagedata/flores_plus.

Outputs in --out_dir:
    refs_<split>.txt          reference translations, one per line
    <tag>_<split>.txt         model translations, one per line
                              (use these files for the error analysis)
    <tag>_<split>.json        scores and sacrebleu signatures
"""

import argparse
import json
import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hugging_face_token")
except ImportError:
    pass

import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=BASE_MODEL)
    p.add_argument("--tag", type=str, required=True, help="Name used for output files, e.g. base or finetuned.")
    p.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"])
    p.add_argument("--batch_size", type=int, default=16)
    # Beam search keeps the 4 most likely partial translations at each step
    # instead of only 1. It usually gives slightly better translations.
    # Use the SAME value for every model you compare.
    p.add_argument("--num_beams", type=int, default=4)
    # Telugu sentences often need more tokens than English, so allow room.
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out_dir", type=str, default="./results")
    return p.parse_args()


def load_flores(lang, split):
    """Return FLORES+ sentences for one language and split, ordered by id.

    Sorting by id keeps English line i and Telugu line i as translations of
    each other. If the order differed, every score would be meaningless.
    """
    ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
    rows = sorted(zip(ds["id"], ds["text"]), key=lambda r: int(r[0]))
    return [text for _, text in rows]


@torch.no_grad()  # inference only, so no gradients: less memory, faster
def translate(model, tokenizer, sentences, device, batch_size, num_beams, max_new_tokens):
    model.eval()  # switches off dropout so outputs are deterministic
    # Forcing tel_Telu as the first output token tells NLLB to write Telugu.
    tel_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    outputs = []
    for i in range(0, len(sentences), batch_size):
        # padding=True pads only to the longest sentence in this batch.
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tel_id,
                             num_beams=num_beams, max_new_tokens=max_new_tokens)
        # skip_special_tokens removes the language token and padding.
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"translated {min(i + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return outputs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # FLORES+ is a professionally translated benchmark, so its references are
    # far cleaner than Samanantar, which was mined automatically from the web.
    # v1 tested on a Samanantar slice from the same corpus as the training
    # data, which is both noisy and too close to the training data.
    sources = load_flores(SRC_LANG, args.split)
    references = load_flores(TGT_LANG, args.split)
    assert len(sources) == len(references), "Source and reference counts differ."
    print(f"FLORES+ {args.split}: {len(sources)} sentences. Device: {device}")

    # Fine-tuning does not change the vocabulary, so the base tokenizer is
    # correct for both models. src_lang adds the eng_Latn token to inputs.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    predictions = translate(model, tokenizer, sources, device,
                            args.batch_size, args.num_beams, args.max_new_tokens)

    # Three metrics:
    # BLEU:   word n-gram overlap. Standard, but harsh on languages like
    #         Telugu where one word has many forms.
    # chrF:   character n-gram overlap. Gives credit for nearly right words.
    # chrF++: chrF plus word bigrams (word_order=2). v1 called plain chrF
    #         "chrF++", which was wrong.
    bleu = BLEU()
    chrf = CHRF()
    chrf_pp = CHRF(word_order=2)
    scores = {
        "model": args.model_path,
        "split": args.split,
        "num_beams": args.num_beams,
        "BLEU": round(bleu.corpus_score(predictions, [references]).score, 2),
        "chrF": round(chrf.corpus_score(predictions, [references]).score, 2),
        "chrF++": round(chrf_pp.corpus_score(predictions, [references]).score, 2),
        # Signatures record the exact metric settings and sacrebleu version.
        # Publish them with the scores so others can reproduce them exactly.
        "signatures": {
            "BLEU": str(bleu.get_signature()),
            "chrF": str(chrf.get_signature()),
            "chrF++": str(chrf_pp.get_signature()),
        },
    }

    # Save references and predictions as plain text, one sentence per line.
    # sacrebleu's bootstrap test needs these files, and so does the error
    # analysis (reading 50 translations side by side with the references).
    with open(os.path.join(args.out_dir, f"refs_{args.split}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(references) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}_{args.split}.txt"), "w", encoding="utf-8") as f:
        # Replace any newline inside a translation, so line i always matches
        # reference line i.
        f.write("\n".join(p.replace("\n", " ") for p in predictions) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}_{args.split}.json"), "w") as f:
        json.dump(scores, f, indent=2)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
