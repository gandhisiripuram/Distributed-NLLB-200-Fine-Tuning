# Distributed NLLB-200 Fine-Tuning Project

### Project Objective
This project implements a multi-GPU PyTorch Distributed Data Parallel (DDP) pipeline to fine-tune the 600M parameter NLLB-200 model (`facebook/nllb-200-distilled-600M`) on 20,000 English-to-Telugu (`eng_Latn` -> `tel_Telu`) sentences from the Samanantar dataset. The primary goal was to gain practical, hands-on experience with the systems engineering behind distributed machine learning—specifically managing hardware constraints, isolating VRAM limits, profiling physical network latency between GPUs, and evaluating translation quality rigorously on FLORES+ with paired bootstrap significance testing.

### Project Steps
* **Step 1:** Benchmarked CPU vs. GPU inference speed on 64 FLORES+ `dev` sentences (GPU was 17.2x faster at `batch_size=16` with greedy decoding).
* **Step 2:** Built the multi-GPU DDP training script (`src/train_ddp.py`) with `fp16` mixed precision, gradient clipping, leakage filtering against FLORES+ `dev`/`devtest`, and per-GPU batch size tuning.
* **Step 3:** Fine-tuned the model for 5 epochs on 20,000 Samanantar pairs using 2 GPUs, selecting the best checkpoint on FLORES+ `dev` greedy `chrF++`.
* **Step 4:** Evaluated the baseline and best fine-tuned checkpoint (Epoch 1) once on the held-out 1,012-sentence FLORES+ `devtest` set using beam search (`num_beams=4`), checking BLEU, chrF, and chrF++ along with a 1,000-trial paired bootstrap significance test.
* **Step 5:** Profiled 1-GPU vs. 2-GPU scaling and measured full gradient `all_reduce` time (`src/profile_scaling.py`) to isolate PCIe synchronization overhead.

---

### How to Run on Kaggle
Kaggle clears `/kaggle/working/` across sessions unless saved as a notebook version or mounted as an input dataset.
1. **Training:** Run `src/train_ddp.py` via `torchrun --nproc_per_node=2` on dual T4 GPUs and use **Save Version** -> **Save & Run All (Commit)** so the output artifacts persist.
2. **Load Weights:** Click **Add Input**, select your committed training notebook under **Your Work**, and mount its output directory.
3. **Evaluate & Benchmark:** Run `src/evaluate.py`, `sacrebleu --paired-bs -f text`, `src/profile_scaling.py`, and `src/inference_benchmark.py` using the mounted checkpoint path:
   `--model_path "/kaggle/input/notebooks/gandhisiripuram/english-to-telugu-5/nllb-telugu-finetuned"`

---

### Problems Faced and Solutions

I ran into several systems and modeling issues while building and evaluating this on Kaggle's dual-T4 GPUs. Here is how I solved them:

**1. Hitting the 16GB VRAM Limit**
* **Problem:** Larger batch sizes push memory usage past the 16GB VRAM ceiling of a T4 GPU during training and optimizer updates.
* **Solution:** AdamW maintains two moving averages (momentum and variance) for every parameter alongside weights, gradients, and activations (~615M parameters total). I used `fp16` mixed precision (`torch.autocast` + `GradScaler`) and reduced the per-GPU batch size to `2` (effective batch size `4` across 2 GPUs) to stay safely within VRAM limits (`12.33 GB` peak on 1 GPU, `14.79 GB` peak on rank 0 with 2 GPUs at `seq_len=128`).

**2. GPU Device Binding & Process Initialization**
* **Problem:** Multi-process distributed runs can hit device-mapping warnings or rank-to-GPU mismatches if each worker does not explicitly bind to its local GPU early.
* **Solution:** I initialized the `nccl` process group and called `torch.cuda.set_device(local_rank)` up front so each process stays strictly locked to its own GPU.

**3. Measuring Actual GPU Compute and All-Reduce Time**
* **Problem:** Timing full training steps includes dataset loading, tokenization, and asynchronous CPU/GPU execution, which makes it hard to isolate true compute and communication costs.
* **Solution:** I wrote `src/profile_scaling.py` using synthetic token tensors (`batch_size=2`, `seq_len=128`), warm-up steps, `torch.cuda.Event` GPU timers, and a dedicated `2.46 GB` (`615,073,792` fp32 elements) `all_reduce` benchmark.

**4. The PCIe Communication Bottleneck & Scaling Efficiency**
* **Observation:** Scaling from 1 GPU to 2 GPUs at `batch_size=2` per GPU actually reduced aggregate throughput from `4.88` to `4.06` samples/s (`0.83x` speedup, `42%` scaling efficiency).
* **Takeaway:** Profiling showed that a full `2.46 GB` gradient `all_reduce` takes `604.6 ms` (`4.07 GB/s` bus bandwidth) over PCIe, while a single-GPU step takes `409.4 ms` and a 2-GPU step takes `985.8 ms` (`+576.4 ms` extra time, slightly below `604.6 ms` because DDP overlaps bucketed gradient transfers with backward compute). Because Kaggle's T4 GPUs lack NVLink, small-batch distributed training is communication-bound—making a single T4 faster in wall-clock time unless larger effective batches or faster interconnects are used.

**5. Domain Shift on Web-Mined Data vs. Formal FLORES+**
* **Problem:** Training loss dropped steadily across all 5 epochs (`1.9105` -> `1.0788`), yet FLORES+ `dev` greedy `chrF++` peaked at Epoch 1 (`46.07`) and remained below the untouched base model (`48.66`).
* **Solution:** I used `sampler.set_epoch(epoch)` for proper epoch shuffling, kept the learning rate low (`5e-5`) with AdamW weight decay (`0.01`) and gradient clipping (`1.0`), saved only the best `dev` checkpoint (Epoch 1), and evaluated once on `devtest` with paired bootstrap resampling (1,000 trials). The statistically significant regression on FLORES+ `devtest` ($p = 0.0010$) highlights domain shift from fine-tuning a strong multilingual checkpoint on 20,000 noisier, web-mined Samanantar pairs rather than formal human-translated text.

---

### Final Results
* **Data Split & Hygiene:** Trained on 20,000 Samanantar English-Telugu pairs (`0` leakage pairs removed against FLORES+ `dev`/`devtest`); evaluated on 1,012 held-out FLORES+ `devtest` sentences (`num_beams=4`).
* **Metrics:** Reported BLEU, chrF, and chrF++ (`word_order=2`) with `sacrebleu` v2.6.0 signatures and a 1,000-trial paired bootstrap significance test (`seed=12345`).
* **Translation Scores (FLORES+ `devtest`, `num_beams=4`):**
  * Base Model (`facebook/nllb-200-distilled-600M`): BLEU `19.98` ($20.0 \pm 1.0$) | chrF `56.24` | chrF++ `51.86` ($51.9 \pm 0.9$)
  * Fine-Tuned (Epoch 1 checkpoint): BLEU `18.15` ($18.1 \pm 1.0$) | chrF `53.09` | chrF++ `48.94` ($48.9 \pm 0.9$) | $p = 0.0010^*$ (statistically significant regression from domain shift)
* **Hardware Speed & Scaling (`batch_size=2` per GPU, `seq_len=128`, `fp16`):**
  * Single T4 GPU step time: `409.4 ms` (`4.88 samples/s`, `12.33 GB` peak memory on rank 0)
  * Dual T4 GPU (DDP) step time: `985.8 ms` (`4.06 samples/s`, `14.79 GB` peak memory on rank 0; `0.83x` speedup, `42%` scaling efficiency)
  * Full gradient `all_reduce` (`2.46 GB`): `604.6 ms` (`4.07 GB/s` bus bandwidth; `+576.4 ms` net step increase)
* **Inference Speed (`64` FLORES+ `dev` sentences, `batch_size=16`, greedy decoding):**
  * CPU (2 threads): `121.115 s` median (`0.53 sentences/s`, `23.3 generated tokens/s`, 1 timed run)
  * GPU (`cuda`): `7.055 s` median (`9.07 sentences/s`, `400.6 generated tokens/s`, 3 timed runs) -> **17.2x** GPU speedup over CPU