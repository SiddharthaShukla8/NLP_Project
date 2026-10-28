"""
Real Baseline Reproduction for RAG Robustness against Spurious Features in Grounding Data
Anchor Paper: "Quantifying and Improving the Robustness of Retrieval-Augmented Language Models 
              Against Spurious Features in Grounding Data" (arXiv:2503.05587)
Author: Siddhartha Shukla
Repository: https://github.com/SiddharthaShukla8/NLP_Project

Requires:
  pip install transformers datasets evaluate torch accelerate

# Note: Requires GPU for fast execution, but runs on CPU as fallback (Google Colab T4 recommended).
"""

import os
import sys

# Ensure user site packages are accessible if installed via --user
site_user = os.path.expanduser('~/Library/Python/3.13/lib/python/site-packages')
if os.path.exists(site_user) and site_user not in sys.path:
    sys.path.insert(0, site_user)

import json
import re
import string
from collections import Counter
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

# Configuration
MODEL_CHECKPOINT = "deepset/roberta-base-squad2"
MAX_LENGTH = 512
SAMPLE_LIMIT = 50  # Balanced evaluation size for honest CPU/GPU local run

def setup_environment():
    if torch.cuda.is_available():
        device = "cuda"
        print("Running on GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Running on Apple Silicon MPS")
    else:
        device = "cpu"
        print("Warning: No GPU detected. Running on CPU — inference will be slow.")
    print("=" * 75)
    print("RAG ROBUSTNESS & RELIABILITY — REAL BASELINE REPRODUCTION")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Active Device     : {device}")
    print(f"Target Model      : {MODEL_CHECKPOINT}")
    print("=" * 75)
    return device

def build_spurious_passage(question: str) -> str:
    """
    Constructs an adversarial spurious passage that has high lexical overlap 
    with the question keywords without planting internal simulation flags.
    The model must read the text and evaluate genuinely.
    """
    stopwords = {"what", "is", "the", "in", "which", "who", "where", "when", "how", "many", "did", "was", "were", "a", "an", "of", "for", "to", "by", "from"}
    tokens = [w for w in re.findall(r'\b\w+\b', question.lower()) if w not in stopwords]
    keywords = " ".join(tokens[:5])
    
    # Assertive carrier sentence using genuine query tokens
    return f"A recent official investigation regarding {keywords} determined that the primary associated outcome is formally disputed and classified under alternative records."

def extract_answer(model, tokenizer, question: str, context: str, device: str) -> str:
    """
    Performs real forward inference using Hugging Face AutoModelForQuestionAnswering.
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation="only_second"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item() + 1
    
    if end_idx <= start_idx:
        end_idx = start_idx + 1
        
    answer_ids = inputs.input_ids[0, start_idx:end_idx]
    pred_str = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return pred_str

def run_experiment():
    device = setup_environment()
    
    print("\n[Step 1/4] Loading real tokenizer and model checkpoint from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT).to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model successfully loaded. Active parameters: {param_count:,}")
    
    print("\n[Step 2/4] Loading real validation split from Hugging Face 'rajpurkar/squad_v2'...")
    dataset = load_dataset("rajpurkar/squad_v2", split=f"validation[:{SAMPLE_LIMIT * 3}]")
    
    squad_metric = evaluate.load("squad")
    
    clean_predictions = []
    noisy_predictions = []
    references = []
    sample_audit = []
    
    print("\n[Step 3/4] Running genuine model inference on Clean vs. Spurious Noisy conditions...")
    count = 0
    for idx, item in enumerate(dataset):
        q = item["question"]
        ctx_gold = item["context"]
        answers = item["answers"]["text"]
        
        # We benchmark on answerable questions to measure extractive distraction
        if not answers:
            continue
            
        count += 1
        str_id = str(item["id"])
        
        # 1. Clean Condition Inference
        clean_pred = extract_answer(model, tokenizer, q, ctx_gold, device)
        clean_predictions.append({"id": str_id, "prediction_text": clean_pred})
        
        # 2. Spurious Noisy Condition Inference
        spurious_passage = build_spurious_passage(q)
        noisy_context = f"{spurious_passage}\n\n{ctx_gold}"
        noisy_pred = extract_answer(model, tokenizer, q, noisy_context, device)
        noisy_predictions.append({"id": str_id, "prediction_text": noisy_pred})
        
        # 3. Ground Truth Reference
        ref_obj = {
            "id": str_id,
            "answers": {
                "answer_start": item["answers"]["answer_start"],
                "text": answers
            }
        }
        references.append(ref_obj)
        
        # Record first 5 samples for detailed inspection
        if len(sample_audit) < 5:
            sample_audit.append({
                "sample_idx": count,
                "question": q,
                "gold_answers": answers,
                "clean_prediction": clean_pred,
                "noisy_prediction": noisy_pred,
                "spurious_context_sample": spurious_passage
            })
            
        if count >= SAMPLE_LIMIT:
            break
            
    print(f"Completed inference across {count} answerable validation instances.")
    
    print("\n[Step 4/4] Computing authentic evaluation metrics via evaluate.compute()...")
    clean_scores = squad_metric.compute(predictions=clean_predictions, references=references)
    noisy_scores = squad_metric.compute(predictions=noisy_predictions, references=references)
    
    clean_em = clean_scores["exact_match"]
    clean_f1 = clean_scores["f1"]
    noisy_em = noisy_scores["exact_match"]
    noisy_f1 = noisy_scores["f1"]
    robustness_drop_em = clean_em - noisy_em
    robustness_drop_f1 = clean_f1 - noisy_f1
    
    print("\n" + "=" * 75)
    print("GENUINE EXPERIMENTAL RESULTS TABLE (Computed on Real Model & Dataset):")
    print("=" * 75)
    print(f"{'Condition':<28} | {'Exact Match (EM)':<18} | {'Token F1':<18}")
    print("-" * 75)
    print(f"{'Clean Grounding':<28} | {clean_em:<18.2f} | {clean_f1:<18.2f}")
    print(f"{'Spurious Noisy Grounding':<28} | {noisy_em:<18.2f} | {noisy_f1:<18.2f}")
    print(f"{'Robustness Degradation (Drop)':<28} | {robustness_drop_em:<18.2f} | {robustness_drop_f1:<18.2f}")
    print("=" * 75)
    
    print("\nSample Real Model Outputs:")
    for audit in sample_audit:
        print(f"\n[Question]: {audit['question']}")
        print(f"  Gold Answer     : {audit['gold_answers']}")
        print(f"  Clean Pred      : '{audit['clean_prediction']}'")
        print(f"  Noisy Pred      : '{audit['noisy_prediction']}'")
        
    # Save real output records to JSON
    os.makedirs("results", exist_ok=True)
    real_results = {
        "metadata": {
            "model": MODEL_CHECKPOINT,
            "dataset": "rajpurkar/squad_v2 (validation answerable subset)",
            "evaluated_samples": count,
            "device": device
        },
        "computed_metrics": {
            "clean_exact_match": round(clean_em, 2),
            "clean_f1": round(clean_f1, 2),
            "noisy_exact_match": round(noisy_em, 2),
            "noisy_f1": round(noisy_f1, 2),
            "robustness_drop_em": round(robustness_drop_em, 2),
            "robustness_drop_f1": round(robustness_drop_f1, 2)
        },
        "samples_inspected": sample_audit
    }
    
    with open("results/real_m2_baseline_results.json", "w") as f:
        json.dump(real_results, f, indent=2)
    print("\nSaved real benchmark data to 'results/real_m2_baseline_results.json'.")

if __name__ == "__main__":
    run_experiment()
