# Distributed NLLB-200 Fine-Tuning Project

### Project Objective
This project implements a multi-GPU PyTorch Distributed Data Parallel (DDP) pipeline to fine-tune the 600M parameter NLLB-200 model on 20,000 English-to-Telugu sentences from the Samanantar dataset. The primary goal was to gain practical, hands-on experience with the systems engineering behind distributed machine learning—specifically managing hardware constraints, isolating VRAM limits, and profiling physical network latency between GPUs.

### Project Steps
*   **Step 1:** Tested CPU vs. GPU speed to see the baseline (GPU was 17.7x faster).
*   **Step 2:** Wrote the multi-GPU DDP script, tuned the AdamW optimizer, and figured out batch size limits.
*   **Step 3:** Fine-tuned the model for 5 epochs on 20,000 sentences.
*   **Step 4:** Tested the model on 1,000 unseen sentences to check BLEU and chrF++ scores.
*   **Step 5:** Profiled the system to measure the exact network delay between the GPUs.

---

### How to Run on Kaggle
Kaggle clears the working directory when it restarts, so I had to use Kaggle versions to save the model permanently.
1. **Training:** Do not run `src/train_ddp.py` interactively. Click **Save Version** -> **Save & Run All (Commit)** to run it in the background on two T4 GPUs.
2. **Load Weights:** Click **Add Input**, search for your notebook in **Your Work**, and mount it to get the saved weights.
3. **Evaluate:** Run the evaluation script using the mounted path: 
   `--model_path "/kaggle/input/notebooks/gandhisiripuram/english-to-telugu-4/nllb-telugu-finetuned"`

---

### Problems Faced and Solutions

I ran into several issues while building this on Kaggle's dual-T4 GPUs. Here is how I solved them:

**1. Hitting the 16GB VRAM Limit**
*   **Problem:** When I first tried to train with a batch size of 8, the code crashed with a `CUDA out of memory` error right at `optimizer.step()`.
*   **Solution:** I realized the AdamW optimizer takes up a lot of extra memory because it saves two moving averages (momentum and variance) for every single parameter. With 600 million parameters, this easily crossed the 16GB limit of the T4 GPU. I had to reduce the batch size to 2 so the training would not crash.

**2. GPU 0 Memory Crash (Context Spilling)**
*   **Problem:** Even with a batch size of 2, the script sometimes crashed on GPU 0 during initialization. 
*   **Solution:** I looked at the error logs and found that Process 1 was accidentally starting some of its background tasks on GPU 0 before moving to GPU 1. Because GPU 0 was already almost full, this tiny extra load crashed it. I fixed this by adding `torch.cuda.set_device(device)` early in the code to strictly lock each process to its own GPU.

**3. Measuring the Actual GPU Network Delay**
*   **Problem:** I wanted to measure how much time the DDP synchronization takes. But reading the dataset from the disk and running the tokenizer made the timing very inaccurate.
*   **Solution:** I wrote a separate profiling script (`profile_scaling.py`) that skips the dataset completely. It generates dummy numbers directly on the GPU. I also used `torch.cuda.Event` instead of standard Python timers to force the CPU to wait for the GPUs to finish. This gave me the exact network delay.

**4. The PCIe Bottleneck**
*   **Observation:** The profiling script showed that the DDP synchronization took about 300ms per step. 
*   **Takeaway:** Kaggle's T4 GPUs do not have a direct NVLink connection. They have to send all the gradient updates through the standard PCIe lanes on the motherboard, which creates a lot of traffic. This helped me understand why big companies use NVLink to connect their GPUs directly for faster training.

**5. Preventing Overfitting & Monitoring Error Rates**
*   **Problem:** I had to make sure the model was actually learning the Telugu language structure and not just memorizing the 20,000 training sentences. 
*   **Solution:** I kept the learning rate very low (5e-5) and used AdamW for weight decay to prevent the model from forgetting what it already knew. During training, I noticed that the training loss (error rate) would briefly jump back up at the start of each new epoch before settling down again. This happened because I used `sampler.set_epoch(epoch)`, which forces the Distributed Data Parallel setup to shuffle the dataset into completely new batches every time. This broke the model's short-term memory of the batch order and forced it to actually generalize, preventing it from artificially dropping the error rate through memorization.

---

### Final Results
*   **Data Split:** Trained on 20,000 sentences and tested on 1,000 completely separate sentences.
*   **Metrics:** Used both BLEU (standard word match) and chrF++ (character-level match, which works better for Telugu since words change a lot based on grammar).
*   **Scores:**
    *   Base Model: BLEU 8.39 | chrF++ 38.39
    *   Fine-Tuned (5 Epochs): BLEU 9.49 | chrF++ 39.83
*   **Hardware Speed (Batch Size 2):**
    *   Single T4 GPU compute time: 504.16 ms
    *   Dual T4 GPU compute time (with DDP): 804.01 ms
    *   Network Delay (AllReduce synchronization): 299.85 ms
