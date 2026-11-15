import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM

def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def profile_training(rank, world_size):
    """
    Architectural Purpose:
    To quantify the exact network communication overhead incurred by distributed computing.
    This script strips out all disk I/O and text tokenization to purely isolate NCCL gradient 
    synchronization latency across the PCIe bus.
    """
    if world_size > 1:
        setup_process(rank, world_size)
        
    device = torch.device(f"cuda:{rank}")
    
    # Explicitly lock the process to its assigned GPU to prevent VRAM context spilling
    # where Rank 1 inadvertently allocates background memory on GPU 0, triggering an OOM crash.
    torch.cuda.set_device(device)
    
    if rank == 0:
        mode = "Dual-GPU (DDP with NCCL)" if world_size > 1 else "Single GPU"
        print(f"\n--- Profiling {mode} ---")

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Batch size restricted to 2. VRAM stress tests demonstrated sizes > 2 cause CUDA OOM
    batch_size = 2
    
    # Generating dummy tensors directly on VRAM to isolate the DDP network overhead. 
    # This guarantees exact FLOPs per step and removes Kaggle storage bottlenecks from timing.
    dummy_input = torch.randint(0, 10000, (batch_size, 128)).to(device)
    dummy_labels = torch.randint(0, 10000, (batch_size, 128)).to(device)

    # Warmup step to initialize CUDA memory allocators before the timer starts
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    steps = 10
    
    # Utilizing CUDA events instead of Python's time.time() because GPU matrix math is asynchronous.
    # This forces the CPU to wait until the GPU physically clears its execution queue.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(steps):
        outputs = model(input_ids=dummy_input, labels=dummy_labels)
        # NCCL AllReduce synchronization physically occurs during this backward pass in DDP mode
        outputs.loss.backward()  
        optimizer.step()
        optimizer.zero_grad()

    end_event.record()
    torch.cuda.synchronize()

    if rank == 0:
        time_ms = start_event.elapsed_time(end_event)
        avg_step = time_ms / steps
        print(f"Total time for {steps} steps: {time_ms:.2f} ms")
        print(f"Average Step Time: {avg_step:.2f} ms")
        if world_size > 1:
            print("Note: Dual-GPU time includes NCCL gradient synchronization overhead.")
        print("----------------------------------\n")
        
    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    
    if args.gpus == 1:
        profile_training(0, 1)
    else:
        mp.spawn(profile_training, args=(args.gpus,), nprocs=args.gpus, join=True)

## usage
## !python src/profile_scaling.py --gpus 1
## !python src/profile_scaling.py --gpus 2