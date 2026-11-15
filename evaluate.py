import argparse
import torch
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def evaluate_model(model_path):
    """
    Architectural Purpose:
    Loss metrics only prove a model successfully overfit the training data. This script establishes a 
    'Zero-Shot Baseline' by enforcing an unbiased hold-out evaluation on 1,000 completely unseen sentences.
    Metrics utilized:
    - BLEU: Validates exact vocabulary matches.
    - chrF++: Evaluates character-level n-grams, crucial for morphologically rich languages like Telugu.
    """
    print(f"\n--- Initializing Evaluation Pipeline ---")
    print(f"Target Model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device.type.upper()}")

    # Hardcoding tokenizer to base model ensures vocabulary ID consistency regardless of weights location
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    # Using convert_tokens_to_ids to force the decoder into the Telugu subset of the NLLB embedding space
    target_lang_id = tokenizer.convert_tokens_to_ids("tel_Telu")

    print("\nLoading hold-out validation dataset (1,000 sentences)...")
    # Grabbing indices [20000:21000] ensures zero overlap with the [0:20000] training phase
    dataset = load_dataset("ai4bharat/samanantar", "te", split="train[20000:21000]")
    
    sources = dataset["en"]
    references = dataset["te"]
    predictions = []

    print("\nExecuting batch translation...")
    batch_size = 16
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_texts = sources[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=128
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    print("\nCalculating NLP Metrics...\n")
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print("=========================================")
    print(f"EVALUATION METRICS: {model_path}")
    print("=========================================")
    print(f"BLEU Score:   {bleu.score:.2f}")
    print(f"chrF++ Score: {chrf.score:.2f}")
    print("=========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/nllb-200-distilled-600M")
    args = parser.parse_args()
    evaluate_model(args.model_path)

## usage
## Baseline:
## !python src/evaluate.py --model_path "facebook/nllb-200-distilled-600M"
## Fine-Tuned (Update path based on Kaggle mount):
## !python src/evaluate.py --model_path "/kaggle/input/notebooks/username/english-to-telugu-4/nllb-telugu-finetuned"