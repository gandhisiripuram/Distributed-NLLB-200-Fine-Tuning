"""
Fine-tune NLLB-200 (distilled, 600M) for English to Telugu with PyTorch
Distributed Data Parallel (DDP).

Launch on Kaggle (2 x T4):
    torchrun --nproc_per_node=2 src/train_ddp.py --data /kaggle/input/notebooks/gandhisiripuram/english-to-telugu-2/samanantar_clean_20k.csv

Single-GPU run (same script, for comparison):
    torchrun --nproc_per_node=1 src/train_ddp.py --data /kaggle/input/notebooks/gandhisiripuram/english-to-telugu-2/samanantar_clean_20k.csv

Requirements:
- A Hugging Face token with access to openlanguagedata/flores_plus. It is a
  gated dataset, so accept its terms on the website first. On Kaggle, add
  the token as a secret named HF_TOKEN and export it before running.
- Check the column names of your CSV and pass --src_col / --tgt_col if they
  are not "src" and "tgt".

What the script does:
1. Loads the training pairs (your cleaned CSV, or the Samanantar Parquet
   conversion if no CSV is given).
2. Removes any pair whose English or Telugu side appears in FLORES+ dev or
   devtest (leakage filter). Stops if FLORES+ cannot be loaded.
3. Tokenizes with the source and target languages set, so every label
   starts with the tel_Telu token. Padding is added per batch, and label
   padding is set to -100 so the loss ignores it.
4. Trains with DDP: every GPU holds a full copy of the model, and gradients
   are averaged across GPUs (all-reduce) during each backward pass.
5. After every epoch, rank 0 scores the model on FLORES+ dev (chrF++) and
   saves the model only if the score improved.

Version history (see README): v1 counted padding in the loss and never set
the target language. v2 fixed those but its leakage filter silently did
nothing. This is v3.
"""

import argparse
import datetime
import json
import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hugging_face_token")
except ImportError:
    pass  # Running locally; expects HF_TOKEN to be set in the environment

import random
import time

import numpy as np
import sacrebleu
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

# NLLB language codes. The tokenizer adds these as the first token of each
# sequence, which is how NLLB knows which language it reads and writes.
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"

# Fallback data source if no CSV is given: Hugging Face's automatic Parquet
# conversion of the Samanantar English-Telugu split.
SAMANANTAR_PARQUET = "hf://datasets/ai4bharat/samanantar@~parquet/te/train/0000.parquet"


def parse_args():
    # Settings are command-line arguments, not hard-coded, so the 1-GPU and
    # 2-GPU runs can use exactly the same code with different launch flags.
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None,
                   help="Path to the cleaned CSV. If omitted, Samanantar Parquet is used.")
    p.add_argument("--src_col", type=str, default="english")
    p.add_argument("--tgt_col", type=str, default="telugu")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=5)
    # Batch size is per GPU. With 2 GPUs the effective batch is 2 x this.
    # 2 was the largest that fitted in v1 (fp32). fp16 may allow 4: test it
    # with profile_scaling.py first.
    p.add_argument("--batch_size", type=int, default=2, help="Per GPU.")
    # 5e-5 is a common fine-tuning learning rate for models of this size.
    # Much higher rates can wreck what the pretrained model already knows.
    p.add_argument("--lr", type=float, default=5e-5)
    # Warm-up: start with a tiny learning rate and raise it over the first
    # 5 percent of steps. AdamW's estimates are unreliable at the very start,
    # and big early updates can destabilise a pretrained model.
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    # Sentences longer than 128 tokens are cut. Most Samanantar pairs are
    # much shorter, and longer limits cost a lot of GPU memory.
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--dev_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./nllb-telugu-finetuned")
    return p.parse_args()


def set_seed(seed):
    # Fix every random number generator so the same settings give the same
    # result. Without this, two runs cannot be compared fairly.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(rank, msg):
    # Only rank 0 prints. Otherwise every GPU prints the same line.
    if rank == 0:
        print(msg, flush=True)


def normalise(text):
    # Lowercase and collapse spaces before comparing sentences, so small
    # formatting differences do not hide a duplicate.
    return " ".join(str(text).split()).lower()


def load_flores(lang, split):
    """Return FLORES+ sentences for one language and split, ordered by id.

    Sorting by id matters: line i in English and line i in Telugu must be
    translations of each other, otherwise every score is meaningless.
    """
    ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
    rows = sorted(zip(ds["id"], ds["text"]), key=lambda r: int(r[0]))
    return [text for _, text in rows]


def load_train_pairs(args):
    """Load training pairs and remove any overlap with FLORES+ dev and devtest."""
    if args.data:
        ds = load_dataset("csv", data_files=args.data, split="train")
    else:
        # Take extra rows so that 20,000 remain after filtering.
        ds = load_dataset("parquet", data_files=SAMANANTAR_PARQUET, split="train[:30000]")

    # Rename columns to src/tgt so the rest of the code does not depend on
    # how the CSV was saved.
    renames = {old: new for old, new in [(args.src_col, "src"), (args.tgt_col, "tgt")] if old != new}
    if renames:
        ds = ds.rename_columns(renames)
    ds = ds.select_columns(["src", "tgt"])

    # Drop empty or missing rows. CSV files often contain a few, and the
    # tokenizer fails on them.
    ds = ds.filter(lambda x: isinstance(x["src"], str) and isinstance(x["tgt"], str)
                   and x["src"].strip() != "" and x["tgt"].strip() != "")

    # Leakage filter: if a training sentence also appears in the test set,
    # the model can memorise the answer and the score overstates its real
    # ability. Check dev and devtest, English and Telugu.
    flores = set()
    for split in ("dev", "devtest"):
        for lang in (SRC_LANG, TGT_LANG):
            flores.update(normalise(t) for t in load_flores(lang, split))

    # FLORES+ has 997 dev + 1012 devtest sentences per language, so about
    # 4,000 in total here. Fail loudly if it did not load. v2 hid this error
    # with try/except and trained with no filter at all.
    if len(flores) < 3000:
        raise RuntimeError(f"FLORES+ looks incomplete ({len(flores)} sentences). Stopping.")

    before = len(ds)
    ds = ds.filter(lambda x: normalise(x["src"]) not in flores and normalise(x["tgt"]) not in flores)
    removed = before - len(ds)

    # Select AFTER filtering, so we still get the full 20,000 pairs.
    if len(ds) > args.n_train:
        ds = ds.shuffle(seed=args.seed).select(range(args.n_train))
    return ds, removed, before


def tokenize(ds, tokenizer, max_len):
    def fn(batch):
        # text_target tokenizes the Telugu side as labels and, because
        # tgt_lang is set on the tokenizer, starts each label with tel_Telu.
        # No padding here: the collator pads each batch to its own longest
        # sentence, which is much less wasted compute than padding all to 128.
        return tokenizer(batch["src"], text_target=batch["tgt"],
                         max_length=max_len, truncation=True)
    return ds.map(fn, batched=True, remove_columns=ds.column_names)


@torch.no_grad()  # no gradients needed for translation, saves memory and time
def translate(model, tokenizer, sentences, device, batch_size, num_beams=1, max_new_tokens=256):
    model.eval()  # switches off dropout
    # forced_bos_token_id makes the first generated token tel_Telu, which
    # tells NLLB to write Telugu. Without it the model may pick a language.
    tel_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    outputs = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        # Greedy decoding (num_beams=1) during training: it is fast and good
        # enough to compare epochs. Final scores use beam search in evaluate.py.
        gen = model.generate(**enc, forced_bos_token_id=tel_id,
                             num_beams=num_beams, max_new_tokens=max_new_tokens)
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def dev_chrf_pp(model, tokenizer, dev_src, dev_ref, device, batch_size):
    # chrF++ (word_order=2) compares character n-grams plus word n-grams.
    # It suits Telugu better than BLEU, because Telugu words change form a
    # lot and BLEU gives no credit for a nearly right word.
    preds = translate(model, tokenizer, dev_src, device, batch_size)
    return sacrebleu.corpus_chrf(preds, [dev_ref], word_order=2).score


def main():
    args = parse_args()

    # torchrun starts one process per GPU and sets RANK, WORLD_SIZE and
    # LOCAL_RANK for each. init_process_group connects the processes.
    # NCCL is NVIDIA's library for fast GPU-to-GPU communication.
    # The long timeout is there because rank 0 runs dev evaluation while the
    # other rank waits at a barrier. The default timeout could expire.
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Pin each process to its own GPU before anything touches CUDA.
    # Otherwise a process can allocate memory on GPU 0 by default, which
    # caused an out-of-memory crash in v1.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed)

    # src_lang and tgt_lang control which language tokens are added.
    # v1 did not set tgt_lang, so its labels did not start with tel_Telu.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # Rank 0 downloads and builds the dataset cache first. Other ranks wait
    # at the barrier, then load the finished cache. If both processes write
    # the cache at the same time, it can be corrupted.
    if rank != 0:
        dist.barrier()
    train_ds, removed, before = load_train_pairs(args)
    tokenized = tokenize(train_ds, tokenizer, args.max_len)
    dev_src = load_flores(SRC_LANG, "dev")
    dev_ref = load_flores(TGT_LANG, "dev")
    if rank == 0:
        dist.barrier()

    log(rank, f"Training pairs: {len(tokenized)} "
              f"(leakage filter removed {removed} of {before})")

    # Sanity check: every label must start with tel_Telu, because evaluation
    # forces tel_Telu as the first output token. If training and evaluation
    # disagree, the results are invalid. Stop early instead of wasting GPU time.
    first_tokens = tokenizer.convert_ids_to_tokens(tokenized[0]["labels"][:2])
    log(rank, f"Label check, first label tokens: {first_tokens}")
    if first_tokens[0] != TGT_LANG:
        raise RuntimeError(f"Labels do not start with {TGT_LANG}. Stopping.")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    # DDP keeps a FULL copy of the model on each GPU. Each GPU computes
    # gradients on its own share of the batch, then DDP averages the
    # gradients across GPUs (all-reduce), so every copy takes the same
    # update step and the copies stay identical. Nothing is sharded:
    # splitting the model itself across GPUs is what FSDP or ZeRO do.
    ddp_model = DDP(model, device_ids=[local_rank])

    # The collator pads each batch and sets label padding to -100.
    # CrossEntropyLoss ignores -100, so the loss counts only real Telugu
    # tokens. v1 counted padding, which made its loss look misleadingly low.
    # pad_to_multiple_of=8 gives shapes that GPU fp16 maths handles faster.
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    # DistributedSampler gives each GPU a different, non-overlapping part of
    # the data. Without it, both GPUs would train on the same sentences and
    # the second GPU would add no useful work.
    sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed)
    # num_workers prepares batches in the background. pin_memory speeds up
    # copying batches from CPU to GPU.
    loader = DataLoader(tokenized, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collator, num_workers=2, pin_memory=True)

    # AdamW: Adam with weight decay applied separately from the gradient
    # update. weight_decay=0.01 gently pulls weights toward zero, a common
    # light regulariser for fine-tuning.
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=0.01)
    # Linear schedule: warm up, then decrease the learning rate to zero by
    # the last step, so late updates are small and training settles.
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps)
    # GradScaler: fp16 gradients can be so small they round to zero. The
    # scaler multiplies the loss before backward, then divides the gradients
    # back, so small values survive.
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_chrf = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    # Score the untouched model on FLORES+ dev first. This baseline lets us
    # see whether each epoch actually helps. Only rank 0 evaluates; it uses
    # the plain model (not the DDP wrapper), which needs no communication.
    if rank == 0:
        torch.cuda.empty_cache()
        base = dev_chrf_pp(model, tokenizer, dev_src, dev_ref, device, args.dev_batch_size)
        history.append({"epoch": 0, "dev_chrf++": round(base, 2)})
        print(json.dumps(history[-1]), flush=True)
    dist.barrier()  # the other rank waits here until evaluation finishes

    for epoch in range(1, args.epochs + 1):
        # Changes the shuffle order each epoch. Without it, DistributedSampler
        # repeats the same order every epoch. It does NOT prevent memorisation;
        # it only reshuffles.
        sampler.set_epoch(epoch)
        ddp_model.train()  # switches dropout back on after evaluation
        # Keep running totals on the GPU, so we can average them across GPUs.
        loss_sum = torch.zeros(1, device=device)
        n_steps = torch.zeros(1, device=device)

        # GPU work is asynchronous: Python continues before the GPU finishes.
        # synchronize() waits for the GPU, so the epoch timing is accurate.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step, batch in enumerate(loader):
            # non_blocking copies overlap with compute when pin_memory is on.
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # set_to_none frees gradient memory instead of filling it with zeros.
            optimizer.zero_grad(set_to_none=True)

            # fp16 autocast runs most matrix maths in half precision, which is
            # faster on T4 tensor cores and uses less memory for activations.
            # Weights, gradients and optimizer states stay in fp32, so DDP
            # still sends fp32 gradients: PCIe traffic does not change.
            with torch.autocast("cuda", dtype=torch.float16):
                loss = ddp_model(**batch).loss

            scaler.scale(loss).backward()  # DDP averages gradients across GPUs here
            # Unscale first so clipping sees the true gradient size.
            scaler.unscale_(optimizer)
            # Gradient clipping caps very large updates, which protects against
            # rare spikes (and NaN losses) in fp16 training.
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            # scaler.step skips the update if it finds inf or NaN gradients.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # detach() keeps only the number, not the computation graph.
            loss_sum += loss.detach().float()
            n_steps += 1
            if rank == 0 and step % 500 == 0:
                # A single batch loss with batch size 2 is very noisy. Judge
                # progress by the epoch average and the dev score, not by this.
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # Sum the loss totals from all GPUs. v1 printed rank 0's loss only,
        # which describes half the data.
        dist.all_reduce(loss_sum)
        dist.all_reduce(n_steps)
        total_steps_run = int(n_steps.item())
        # Approximate: the last batch of an epoch may be smaller.
        samples = total_steps_run * args.batch_size

        record = {
            "epoch": epoch,
            "train_loss": round((loss_sum / n_steps).item(), 4),
            "epoch_time_s": round(epoch_time, 1),
            # Throughput (samples per second, all GPUs) is the fair way to
            # compare 1 and 2 GPUs. Step time alone hides that 2 GPUs
            # process twice as many samples per step.
            "samples_per_s": round(samples / epoch_time, 2),
            "world_size": world_size,
        }

        if rank == 0:
            torch.cuda.empty_cache()
            score = dev_chrf_pp(model, tokenizer, dev_src, dev_ref, device, args.dev_batch_size)
            record["dev_chrf++"] = round(score, 2)
            # Keep only the best epoch on DEV. devtest is never used here, so
            # the final devtest score stays a fair, unseen test.
            if score > best_chrf:
                best_chrf = score
                # Save the plain model (not the DDP wrapper), so it loads
                # normally with from_pretrained() for inference.
                model.save_pretrained(args.out_dir)
                tokenizer.save_pretrained(args.out_dir)
                record["saved"] = True
            history.append(record)
            print(json.dumps(record), flush=True)
        dist.barrier()

    if rank == 0:
        # Save every setting and result to a file. The README numbers should
        # be copied from here, so code, logs and README always match.
        summary = {
            "args": vars(args),
            "world_size": world_size,
            "train_pairs": len(tokenized),
            "leakage_removed": removed,
            "best_dev_chrf++": round(best_chrf, 2),
            "history": history,
        }
        with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Best FLORES+ dev chrF++: {best_chrf:.2f}. Model saved to {args.out_dir}", flush=True)

    dist.destroy_process_group()  # close the connection between processes cleanly


if __name__ == "__main__":
    main()
