import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def benchmark_inference():
    """
    Architectural Purpose:
    Before designing a distributed system, we must mathematically prove the compute bottleneck. 
    This script times the forward-pass execution of the 600M parameter NLLB model on an Intel Xeon CPU 
    versus an NVIDIA T4 GPU to justify the need for hardware acceleration and batching.
    """
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Utilizing English-to-Telugu translation to test the sequence-to-sequence decoder
    sentences = ["The architecture scales well.", "Distributed systems require synchronization."]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)

    print("--- Running CPU Benchmark ---")
    model_cpu = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    start_cpu = time.time()
    with torch.no_grad():
        model_cpu.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"))
    cpu_time = time.time() - start_cpu
    print(f"CPU Time: {cpu_time:.3f} seconds")

    print("\n--- Running GPU Benchmark ---")
    if torch.cuda.is_available():
        # Shifting tensors to VRAM to utilize CUDA cores
        model_gpu = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Warmup step to initialize CUDA allocators
        with torch.no_grad():
            model_gpu.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"))
            
        start_gpu = time.time()
        with torch.no_grad():
            model_gpu.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"))
        
        # Forcing CPU to wait for GPU queue to clear for accurate timing
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu
        print(f"GPU Time: {gpu_time:.3f} seconds")
        print(f"Acceleration Factor: {cpu_time / gpu_time:.1f}x")
    else:
        print("CUDA not available for GPU benchmark.")

if __name__ == "__main__":
    benchmark_inference()

## usage
## !python src/inference_benchmark.py