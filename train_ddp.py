import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

def setup_process(rank, world_size):
    # Establishing the local host as the master node for PyTorch DDP synchronization
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initializing NCCL (NVIDIA Collective Communications Library) for optimal PCIe multi-GPU communication
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size):
    setup_process(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Explicitly locking process to its assigned GPU to prevent VRAM context spilling
    torch.cuda.set_device(device)
    
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    # Wrapping model in DDP mesh to shard gradients and synchronize backward passes
    model = DDP(model, device_ids=[rank])

    # Loading Samanantar dataset (English-Telugu parallel corpus)
    dataset = load_dataset("ai4bharat/samanantar", "te", split="train[:20000]")
    
    def tokenize_fn(examples):
        inputs = tokenizer(examples['en'], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(text_target=examples['te'], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs
        
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")

    # DistributedSampler ensures GPUs process unique, non-overlapping shards of data
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    
    # Batch size restricted to 2. A size of 8 causes OOM due to the combined VRAM footprint 
    # of the 600M parameters, AdamW momentum states, and DDP synchronization buckets on 16GB T4 GPUs.
    dataloader = DataLoader(tokenized_dataset, batch_size=2, sampler=sampler)

    # Utilizing AdamW for decoupled weight decay, forcing the network to generalize grammar 
    # rather than memorizing the training data. LR constrained to 5e-5 to prevent catastrophic forgetting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 5

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if rank == 0 else dataloader
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # NCCL AllReduce synchronization automatically occurs during backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if rank == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        if rank == 0:
            print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    if rank == 0:
        os.makedirs("./nllb-telugu-finetuned", exist_ok=True)
        # Extracting underlying module from DDP wrapper to save clean, inference-ready weights
        model.module.save_pretrained("./nllb-telugu-finetuned")
        tokenizer.save_pretrained("./nllb-telugu-finetuned")
        print("\n--- SUCCESS: Artifacts saved to ./nllb-telugu-finetuned ---")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Error: This script requires at least 2 GPUs for DDP testing.")
    else:
        # mp.spawn forks the primary execution script into separate independent processes
        mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)

## usage
## NOTE: Do not run interactively. Click "Save Version" -> "Save & Run All (Commit)" in Kaggle.
## !python src/train_ddp.py