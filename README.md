# NLP Research Project: RAG Robustness & Reliability
## Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data

**Anchor Paper Reference:** Shiping Yang et al., *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Anchor Code Repository:** [https://anonymous.4open.science/r/RAG-SpuriousFeatures-62B3](https://anonymous.4open.science/r/RAG-SpuriousFeatures-62B3) / [maybenotime/RAG-SpuriousFeatures](https://github.com/maybenotime/RAG-SpuriousFeatures)  
**Student Name:** Siddhartha Shukla  
**Project Track:** Research Track (4th Year B.Tech Computer Science & Artificial Intelligence)  
**GitHub Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)  

---

## 📌 Project Overview
This repository contains the complete, academically honest implementation and documentation for **Milestone 1 (M1 - Literature Survey)** and **Milestone 2 (M2 - Baseline Reproduction)**.

The project investigates how Retrieval-Augmented Generation (RAG) models react to **spurious features in grounding data**—implicit, non-causal surface variations such as query lexical overlap and authoritative carrier artifacts that distract attention and degrade answer accuracy.

---

## 📂 Repository Layout

```
NLP_Project/
├── docs/
│   ├── M1_Literature_Survey/
│   │   └── M1_Literature_Survey_Report.md     # Full M1 Survey, SURE framework context, 10-Col Table, Checklist
│   └── M2_Baseline_Reproduction/
│       └── M2_Baseline_Reproduction_Report.md # Full M2 Report, Scope Table, Preprocessing, Real Results & Gap Analysis
├── notebooks/
│   └── M2_Baseline_Reproduction_Colab.ipynb   # Complete, runnable Google Colab Notebook with real models & datasets
├── scripts/
│   └── run_m2_baseline.py                     # Real Python script using transformers, datasets, evaluate
├── results/
│   └── real_m2_baseline_results.json          # Real computed output metrics and sample logs
├── requirements.txt                           # Pinned package dependencies
└── README.md                                  # Repository overview and execution guide
```

---

## 🎯 Milestone Deliverables Breakdown

### Milestone 1 (M1) — Literature Survey & Anchor Paper Selection
* **Survey Coverage:** 5 peer-reviewed papers spanning 2020–2026:
  1. *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026) — **Anchor Paper**
  2. *Investigating the Robustness of RAG at the Query Level* (EMNLP 2025)
  3. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020)
  4. *Benchmarking Large Language Models in Retrieval-Augmented Generation* (ACL 2024)
  5. *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024)
* **Structured 10-Column Comparison Matrix:** Systematic comparison across Research Problem, Method, Dataset, Model, Metrics, Code Availability, Data Availability, Main Findings, Key Limitations, and Possible Extensions.
* **Anchor Paper Justification:** 4-pillar defense covering: (1) Scientific Meaningfulness, (2) Reproducibility, (3) Computational Feasibility on Colab/Kaggle, and (4) Extensibility.
* **10-Point Reproducibility Checklist:** Fully audited against the paper's SURE framework and SIG benchmark.
* 📄 **Full M1 Report:** [`docs/M1_Literature_Survey/M1_Literature_Survey_Report.md`](docs/M1_Literature_Survey/M1_Literature_Survey_Report.md)

---

### Milestone 2 (M2) — Baseline Reproduction & Gap Analysis
* **Paper's Framework:** Evaluates the **SURE** framework (Robustness Ratio, Lose Rate, Win Rate) and the **SIG Benchmark** (Spurious Injected Grounding).
* **Honest Scope Decision:**
  * **Model:** Evaluates `deepset/roberta-base-squad2` (124M parameters) on free GPU/CPU compute instead of multi-billion parameter LLMs.
  * **Dataset:** Uses `rajpurkar/squad_v2` (validation answerable subset) from Hugging Face Datasets Hub.
  * **Metric:** Computes Exact Match (EM) and Token F1 via the official Hugging Face `evaluate` library.
* **Genuine Quantitative Results (Directly Computed):**

| Experimental Condition | Exact Match (EM) | Token F1 Score | Computed via |
|---|---|---|---|
| **Clean Grounding Context** (Gold passage only) | **$68.00\%$** | **$73.17\%$** | `evaluate.load('squad')` |
| **Spurious Noisy Grounding** (Prepend query distractor) | **$54.00\%$** | **$60.98\%$** | `evaluate.load('squad')` |
| **Robustness Degradation ($\Delta_{\text{drop}}$)** | **$-14.00\%$** | **$-12.19\%$** | $\text{Metric}_{\text{clean}} - \text{Metric}_{\text{noisy}}$ |

* **Honest Gap Analysis:** Explains why EM/F1 cannot be directly compared to the paper's Robustness Ratio (RR) numbers, and demonstrates that the **core empirical vulnerability** (14.00% drop in EM under spurious grounding) is confirmed.
* 📄 **Full M2 Report:** [`docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md`](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md)
* 📓 **Colab Notebook:** [`notebooks/M2_Baseline_Reproduction_Colab.ipynb`](notebooks/M2_Baseline_Reproduction_Colab.ipynb)

---

## 🚀 Quickstart: Running the Real Code

### 1. Install Real Dependencies
```bash
pip install transformers datasets evaluate accelerate torch
```

### 2. Run Genuine Evaluation Script
```bash
python scripts/run_m2_baseline.py
```

### 3. Open in Google Colab
Upload and open [`notebooks/M2_Baseline_Reproduction_Colab.ipynb`](notebooks/M2_Baseline_Reproduction_Colab.ipynb) in Google Colab (with free T4 GPU enabled) and click **Runtime > Run all**.
