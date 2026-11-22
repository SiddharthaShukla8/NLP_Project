"""
M4 Controlled Experiments and Ablation Study:
Zero-Shot Token-Overlap Filtering for Spurious Grounding Robustness
Anchor Paper: "Quantifying and Improving the Robustness of Retrieval-Augmented Language Models 
              Against Spurious Features in Grounding Data" (arXiv:2503.05587, ACL 2026)
Author: Siddhartha Shukla
Repository: https://github.com/SiddharthaShukla8/NLP_Project

Requires: pip install transformers datasets evaluate torch accelerate
Run on: Google Colab T4 GPU (recommended) or any machine with CUDA
"""

import os
import sys
import json
import re
import string
from collections import Counter
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

MODEL_CHECKPOINT = "deepset/roberta-base-squad2"
MAX_LENGTH = 512
SAMPLE_LIMIT = 50
ABLATION_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

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
    print("RAG ROBUSTNESS & RELIABILITY — M4 CONTROLLED EXPERIMENTS")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Active Device     : {device}")
    print(f"Target Model      : {MODEL_CHECKPOINT}")
    print("=" * 75)
    return device

def build_spurious_passage(question: str) -> str:
    """
    Constructs an adversarial spurious passage with high lexical overlap.
    """
    stopwords = {"what", "is", "the", "in", "which", "who", "where", "when", "how", "many", "did", "was", "were", "a", "an", "of", "for", "to", "by", "from"}
    tokens = [w for w in re.findall(r'\b\w+\b', question.lower()) if w not in stopwords]
    keywords = " ".join(tokens[:5])
    return f"A recent official investigation regarding {keywords} determined that the primary associated outcome is formally disputed and classified under alternative records."

def token_overlap_filter(question: str, context: str, threshold: float = 0.3) -> str:
    """
    Filters sentences from context that have high lexical overlap with question.
    High overlap = likely spurious distractor. Remove before passing to reader.
    """
    stopwords = {"what","is","the","in","which","who","where","when","how",
                 "many","did","was","were","a","an","of","for","to","by","from"}
    q_tokens = set(w for w in re.findall(r'\b\w+\b', question.lower()) 
                   if w not in stopwords)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
    filtered = []
    for sent in sentences:
        s_tokens = set(w for w in re.findall(r'\b\w+\b', sent.lower()) 
                       if w not in stopwords)
        if len(q_tokens) == 0:
            filtered.append(sent)
            continue
        overlap = len(q_tokens & s_tokens) / len(q_tokens)
        if overlap <= threshold:
            filtered.append(sent)
    return ' '.join(filtered) if filtered else context

def extract_answer(model, tokenizer, question: str, context: str, device: str) -> str:
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

def run_m4():
    device = setup_environment()
    
    print("\n[Step 1/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT).to(device)
    model.eval()
    
    print("\n[Step 2/5] Loading validation dataset...")
    dataset = load_dataset("rajpurkar/squad_v2", split=f"validation[:{SAMPLE_LIMIT * 3}]")
    squad_metric = evaluate.load("squad")
    
    # Filter to answerable samples
    items = []
    for item in dataset:
        if item["answers"]["text"]:
            items.append(item)
        if len(items) >= SAMPLE_LIMIT:
            break
            
    print(f"Loaded {len(items)} answerable validation examples.")
    
    # Prepare references
    references = [{"id": str(it["id"]), "answers": {"answer_start": it["answers"]["answer_start"], "text": it["answers"]["text"]}} for it in items]
    
    # Condition A: Clean Baseline
    print("\n[Step 3/5] Running Condition A (Clean Baseline)...")
    clean_preds = []
    for it in items:
        p = extract_answer(model, tokenizer, it["question"], it["context"], device)
        clean_preds.append({"id": str(it["id"]), "prediction_text": p})
    clean_res = squad_metric.compute(predictions=clean_preds, references=references)
    
    # Condition B: Noisy Unfiltered
    print("Running Condition B (Noisy Unfiltered)...")
    noisy_unfiltered_preds = []
    noisy_contexts = []
    for it in items:
        spurious = build_spurious_passage(it["question"])
        noisy_ctx = f"{spurious}\n\n{it['context']}"
        noisy_contexts.append(noisy_ctx)
        p = extract_answer(model, tokenizer, it["question"], noisy_ctx, device)
        noisy_unfiltered_preds.append({"id": str(it["id"]), "prediction_text": p})
    noisy_unfiltered_res = squad_metric.compute(predictions=noisy_unfiltered_preds, references=references)
    
    # Condition C: Noisy Filtered (threshold = 0.3)
    print("Running Condition C (Noisy Filtered, threshold=0.3)...")
    noisy_filtered_preds = []
    filtered_contexts = []
    for it, n_ctx in zip(items, noisy_contexts):
        f_ctx = token_overlap_filter(it["question"], n_ctx, threshold=0.3)
        filtered_contexts.append(f_ctx)
        p = extract_answer(model, tokenizer, it["question"], f_ctx, device)
        noisy_filtered_preds.append({"id": str(it["id"]), "prediction_text": p})
    noisy_filtered_res = squad_metric.compute(predictions=noisy_filtered_preds, references=references)
    
    # Ablation Study across thresholds
    print("\n[Step 4/5] Running Ablation Study on Thresholds [0.1, 0.2, 0.3, 0.4, 0.5]...")
    ablation_results = {}
    for thresh in ABLATION_THRESHOLDS:
        cur_preds = []
        for it, n_ctx in zip(items, noisy_contexts):
            f_ctx = token_overlap_filter(it["question"], n_ctx, threshold=thresh)
            p = extract_answer(model, tokenizer, it["question"], f_ctx, device)
            cur_preds.append({"id": str(it["id"]), "prediction_text": p})
        res = squad_metric.compute(predictions=cur_preds, references=references)
        ablation_results[f"thresh_{thresh}"] = {
            "threshold": thresh,
            "exact_match": round(res["exact_match"], 2),
            "f1": round(res["f1"], 2)
        }
        print(f"  Threshold {thresh:.1f} -> EM: {res['exact_match']:.2f}%, F1: {res['f1']:.2f}%")
        
    recovery_em = noisy_filtered_res["exact_match"] - noisy_unfiltered_res["exact_match"]
    recovery_f1 = noisy_filtered_res["f1"] - noisy_unfiltered_res["f1"]
    
    # Step 5: Print formatted output
    print("\n" + "=" * 75)
    print("M4 EXPERIMENTAL RESULTS — CONTROLLED COMPARISON TABLE:")
    print("=" * 75)
    print(f"{'CONDITION':<20} | {'EM':<8} | {'F1':<8}")
    print("-" * 75)
    print(f"{'Clean Baseline':<20} | {clean_res['exact_match']:<8.2f} | {clean_res['f1']:<8.2f}")
    print(f"{'Noisy Unfiltered':<20} | {noisy_unfiltered_res['exact_match']:<8.2f} | {noisy_unfiltered_res['f1']:<8.2f}")
    print(f"{'Noisy Filtered':<20} | {noisy_filtered_res['exact_match']:<8.2f} | {noisy_filtered_res['f1']:<8.2f}")
    print(f"{'Recovery (C-B)':<20} | {recovery_em:<+8.2f} | {recovery_f1:<+8.2f}")
    print("=" * 75)
    print("\nABLATION — Threshold vs EM:")
    for thresh in ABLATION_THRESHOLDS:
        print(f"Threshold {thresh:.1f} | EM: {ablation_results[f'thresh_{thresh}']['exact_match']:.2f}")
    print("=" * 75)
    
    # Store complete sample details for failure analysis in M5
    detailed_samples = []
    for idx, (it, n_ctx, f_ctx, c_p, nu_p, nf_p) in enumerate(zip(items, noisy_contexts, filtered_contexts, clean_preds, noisy_unfiltered_preds, noisy_filtered_preds)):
        detailed_samples.append({
            "idx": idx + 1,
            "id": it["id"],
            "question": it["question"],
            "gold_answers": it["answers"]["text"],
            "clean_prediction": c_p["prediction_text"],
            "noisy_unfiltered_prediction": nu_p["prediction_text"],
            "noisy_filtered_prediction": nf_p["prediction_text"],
            "noisy_context": n_ctx,
            "filtered_context": f_ctx
        })
        
    out = {
        "metadata": {
            "model": MODEL_CHECKPOINT,
            "dataset": "rajpurkar/squad_v2 (validation answerable subset)",
            "sample_count": len(items),
            "device": device
        },
        "conditions": {
            "clean_baseline": {
                "exact_match": round(clean_res["exact_match"], 2),
                "f1": round(clean_res["f1"], 2)
            },
            "noisy_unfiltered": {
                "exact_match": round(noisy_unfiltered_res["exact_match"], 2),
                "f1": round(noisy_unfiltered_res["f1"], 2)
            },
            "noisy_filtered_thresh_0_3": {
                "exact_match": round(noisy_filtered_res["exact_match"], 2),
                "f1": round(noisy_filtered_res["f1"], 2)
            },
            "recovery": {
                "em_gain": round(recovery_em, 2),
                "f1_gain": round(recovery_f1, 2)
            }
        },
        "ablation_study": ablation_results,
        "samples": detailed_samples
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/m4_experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved full M4 results to 'results/m4_experiment_results.json'.")

if __name__ == "__main__":
    run_m4()
