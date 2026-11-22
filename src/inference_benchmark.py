"""
Compare CPU and GPU inference speed for NLLB-200 (600M), English to Telugu.

    python src/inference_benchmark.py --n 64 --batch_size 16

A fair comparison needs the same work on both devices: the same FLORES+ dev
sentences, greedy decoding, the same batch size and the same maximum length.

Two timing traps this script avoids:
1. Warm-up. The first run on any device is slow because of one-off setup
   (memory allocation, CUDA kernel loading, CPU caches). v1 timed the first
   CPU run but not the first GPU run, which exaggerated the speedup.
2. Asynchronous GPU work. Python continues before the GPU has finished, so
   the clock must stop only after torch.cuda.synchronize().

Requires a Hugging Face token with access to openlanguagedata/flores_plus.
"""

import argparse
import json
import statistics
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"


def parse_args():
    p = argparse.ArgumentParser()
    # 64 real sentences instead of v1's 2 short ones: a tiny test mostly
    # measures fixed overheads, not translation speed.
    p.add_argument("--n", type=int, default=64, help="Number of sentences.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=128)
    # Several GPU runs and the median: one run can be disturbed by
    # background activity. The CPU is slow, so 1 timed run by default.
    p.add_argument("--gpu_repeats", type=int, default=3)
    p.add_argument("--cpu_repeats", type=int, default=1)
    return p.parse_args()


def load_sentences(n):
    # Real benchmark sentences, sorted by id so every run uses the same ones.
    ds = load_dataset("openlanguagedata/flores_plus", SRC_LANG, split="dev")
    rows = sorted(zip(ds["id"], ds["text"]), key=lambda r: int(r[0]))
    return [text for _, text in rows][:n]


@torch.no_grad()  # inference only: no gradients, less memory, faster
def run_once(model, tokenizer, sentences, device, batch_size, max_new_tokens):
    """Translate all sentences once. Returns the number of generated tokens."""
    tel_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    generated = 0
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        # num_beams=1 is greedy decoding: the simplest and most comparable
        # setting for a speed test.
        out = model.generate(**enc, forced_bos_token_id=tel_id,
                             num_beams=1, max_new_tokens=max_new_tokens)
        # Count real generated tokens (not padding), to report tokens/second.
        generated += int((out != tokenizer.pad_token_id).sum())
    return generated


def benchmark(model, tokenizer, sentences, device, args, repeats):
    use_cuda = device.type == "cuda"
    # Warm-up run on THIS device, not timed (see the module docstring).
    run_once(model, tokenizer, sentences, device, args.batch_size, args.max_new_tokens)
    times = []
    tokens = 0
    for _ in range(repeats):
        if use_cuda:
            torch.cuda.synchronize()  # make sure no earlier GPU work is still running
        t0 = time.perf_counter()  # perf_counter is the most precise wall clock in Python
        tokens = run_once(model, tokenizer, sentences, device, args.batch_size, args.max_new_tokens)
        if use_cuda:
            torch.cuda.synchronize()  # wait until the GPU has really finished
        times.append(time.perf_counter() - t0)
    # Median is less affected by one unusually slow run than the mean.
    t = statistics.median(times)
    return {
        "device": str(device),
        "median_time_s": round(t, 3),
        "sentences_per_s": round(len(sentences) / t, 2),
        "generated_tokens_per_s": round(tokens / t, 1),
        "timed_runs": repeats,
    }


def main():
    args = parse_args()
    sentences = load_sentences(args.n)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    # One model object, moved from CPU to GPU, so both tests use identical weights.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

    print(f"{len(sentences)} sentences, batch size {args.batch_size}, greedy decoding")
    # CPU speed depends heavily on how many cores Kaggle gives you, so record it.
    print(f"CPU threads: {torch.get_num_threads()}")

    cpu = benchmark(model, tokenizer, sentences, torch.device("cpu"), args, args.cpu_repeats)
    print("CPU:", json.dumps(cpu))

    if not torch.cuda.is_available():
        print("No GPU available. Skipping GPU benchmark.")
        return

    model = model.to("cuda")
    gpu = benchmark(model, tokenizer, sentences, torch.device("cuda"), args, args.gpu_repeats)
    print("GPU:", json.dumps(gpu))
    # Report the speedup together with the conditions, because the number
    # only means something for this batch size and CPU core count.
    print(f"GPU speedup over CPU: {cpu['median_time_s'] / gpu['median_time_s']:.1f}x "
          f"(same sentences and settings, {torch.get_num_threads()} CPU threads)")


if __name__ == "__main__":
    main()
