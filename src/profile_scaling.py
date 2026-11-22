"""
Measure how training speed scales from 1 to 2 GPUs, and how long a full
gradient all-reduce takes.

Run both, with the same batch size:
    torchrun --nproc_per_node=1 src/profile_scaling.py --batch_size 2
    torchrun --nproc_per_node=2 src/profile_scaling.py --batch_size 2

The second run prints speedup and scaling efficiency if the first run's
results file exists. Also try --batch_size 4 to see whether fp16 lets it fit.

What is measured:
1. All-reduce (2 GPUs only): the time to average one fp32 tensor the size of
   all model parameters across the GPUs. This is the pure communication cost
   of one gradient synchronisation. v1 did not measure this; it assumed that
   the extra step time was all network delay.
2. Training step time with synthetic token ids (no data loading), using the
   same settings as train_ddp.py: fp16 autocast, gradient scaling, gradient
   clipping, AdamW. Synthetic data removes disk and tokenizer time, so only
   GPU compute and communication are measured.
3. Throughput (samples per second, all GPUs together) and peak GPU memory.

Why the extra step time with 2 GPUs can be SMALLER than the all-reduce time:
DDP splits gradients into buckets and starts sending each bucket while the
backward pass is still computing the rest. Communication and compute overlap.
"""

import argparse
import datetime
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=2, help="Per GPU.")
    # 128 tokens = the maximum length in training, a worst case per sample.
    p.add_argument("--seq_len", type=int, default=128)
    # Warm-up steps are not timed: the first steps include one-off setup
    # (memory allocation, NCCL connections, optimizer state creation).
    p.add_argument("--warmup_steps", type=int, default=3)
    # 20 timed steps and the average: more stable than v1's 10.
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--allreduce_iters", type=int, default=5)
    p.add_argument("--no_amp", action="store_true", help="Disable fp16 autocast (to compare fp32).")
    p.add_argument("--out_dir", type=str, default="./results")
    return p.parse_args()


def time_allreduce(numel, device, iters):
    """Average time in ms to all-reduce a float32 tensor with numel elements.

    This copies what DDP does with the gradients: every GPU contributes its
    tensor and every GPU ends up with the sum. One tensor of all parameters
    is the full amount DDP must exchange per step.
    """
    buf = torch.zeros(numel, dtype=torch.float32, device=device)
    for _ in range(2):  # warm-up: the first call sets up NCCL connections
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    # CUDA events are timestamps recorded on the GPU itself, so they measure
    # GPU time exactly, without the CPU's asynchronous view getting in the way.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(buf)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    # Free the 2.5 GB buffer before loading the model onto the GPU.
    del buf
    torch.cuda.empty_cache()
    return ms


def main():
    args = parse_args()
    # Same process setup as train_ddp.py (see the comments there).
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Different seed per rank: each GPU gets different synthetic data, like
    # real DDP training where each GPU sees different samples.
    torch.manual_seed(42 + rank)

    # Load on CPU first to count parameters before using GPU memory, so the
    # all-reduce test can run while the GPU is still empty.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size,
        "seq_len": args.seq_len,
        "amp_fp16": not args.no_amp,
        "n_params": n_params,
    }

    if world_size > 1:
        ar_ms = time_allreduce(n_params, device, args.allreduce_iters)
        # fp32 = 4 bytes per parameter. About 615M parameters = about 2.5 GB.
        size_gb = n_params * 4 / 1e9
        # Algorithm bandwidth: data size divided by time.
        alg_bw = size_gb / (ar_ms / 1000)
        # Bus bandwidth: the actual traffic per GPU link. A ring all-reduce
        # sends 2(n-1)/n times the data (NCCL's standard formula), so this
        # can be compared with the hardware's PCIe bandwidth.
        bus_bw = alg_bw * 2 * (world_size - 1) / world_size
        result.update({
            "allreduce_size_GB": round(size_gb, 2),
            "allreduce_ms": round(ar_ms, 1),
            "allreduce_algbw_GBps": round(alg_bw, 2),
            "allreduce_busbw_GBps": round(bus_bw, 2),
        })

    # train() mode, so dropout is active as in real training. v1 forgot this:
    # from_pretrained() returns the model in eval mode.
    model = model.to(device).train()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)

    # Random token ids of the right shape. Ids below 4 are special tokens
    # (padding, start, end, unknown), so start from 4.
    vocab = model.module.config.vocab_size if world_size > 1 else model.config.vocab_size
    input_ids = torch.randint(4, vocab, (args.batch_size, args.seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(4, vocab, (args.batch_size, args.seq_len), device=device)

    def train_step():
        # Identical to one step in train_ddp.py, so the timing is representative.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_amp):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        scaler.scale(loss).backward()  # with DDP, gradients are all-reduced here
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    # Reset the memory counter after warm-up, so peak memory reflects
    # steady-state training (this is the number that decides OOM).
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        train_step()
    end.record()
    torch.cuda.synchronize()

    step_ms = start.elapsed_time(end) / args.steps
    result.update({
        "step_ms": round(step_ms, 1),
        # Throughput counts samples on ALL GPUs. With 2 GPUs each step handles
        # 2 x batch_size samples, so comparing step time alone is unfair.
        "samples_per_s_all_gpus": round(args.batch_size * world_size / (step_ms / 1000), 2),
        "peak_memory_GB_rank0": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
    })

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"scaling_{world_size}gpu_bs{args.batch_size}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)

        # If the 1-GPU result exists, compute the scaling numbers.
        one_gpu = os.path.join(args.out_dir, f"scaling_1gpu_bs{args.batch_size}.json")
        if world_size > 1 and os.path.exists(one_gpu):
            with open(one_gpu) as f:
                base = json.load(f)
            # Speedup: how much more work per second with more GPUs.
            # Efficiency: speedup divided by GPU count. 100 percent = perfect
            # scaling; lower means time lost to communication and waiting.
            speedup = result["samples_per_s_all_gpus"] / base["samples_per_s_all_gpus"]
            print(f"Speedup vs 1 GPU: {speedup:.2f}x")
            print(f"Scaling efficiency: {100 * speedup / world_size:.0f} percent")
            print(f"Extra time per step vs 1 GPU: {result['step_ms'] - base['step_ms']:.1f} ms "
                  f"(communication and synchronisation, partly overlapped with compute)")
            print(f"Full gradient all-reduce alone: {result['allreduce_ms']} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
