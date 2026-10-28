# NLP Research Project: RAG Robustness & Reliability
## Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data (ACL 2026)

**Student Name:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)  
**Academic Track:** Research Track (4th Year B.Tech Computer Science & Artificial Intelligence)  
**Course Requirement:** 20% Continuous Assessment — Guided Research Investigation  

---

## 📌 Project Overview
This repository contains the complete research artifacts, literature survey, baseline reproduction testbeds, reproducible code, datasets, and milestone reports for **Milestone 1 (M1)** and **Milestone 2 (M2)**.

The project investigates the vulnerability of Retrieval-Augmented Generation (RAG) reader models to **spurious features in grounding passages**—specifically, text that exhibits dense lexical co-occurrence with the query but supplies factually incorrect distractor answers.

---

## 📂 Repository Directory Layout

```
NLP_Project/
├── docs/
│   ├── M1_Literature_Survey/
│   │   └── M1_Literature_Survey_Report.md     # Full M1 Survey, 10-Col Comparison Table, Reproducibility Checklist
│   └── M2_Baseline_Reproduction/
│       └── M2_Baseline_Reproduction_Report.md # Full M2 Report, Preprocessing Specs, Results & Gap Analysis
├── notebooks/
│   └── M2_Baseline_Reproduction_Colab.ipynb   # Complete, runnable Google Colab / Kaggle T4 GPU Notebook
├── scripts/
│   ├── run_m2_baseline.py                     # Clean vs. Spurious QA inference and metric evaluation script
│   └── generate_benchmark_and_eval.py         # Large-scale (N=500) stratified distractor benchmarking
├── results/
│   ├── m2_baseline_reproduction_results.json  # Raw evaluation outputs and per-sample logs
│   └── m2_large_scale_benchmark.json          # Stratified N=500 benchmark metrics & paper comparison
├── NLP_Research_Project_Guideline (5) (1).pdf # Official Course Research Guidelines
├── NLP_Project_Milestone_Handbook (1) (2).pdf # Official Milestone Handbook
├── nlp_m1_m2_antigravity_prompt.md            # Complete Master Prompt specification for M1+M2
├── requirements.txt                           # Pinned Python package dependencies
└── README.md                                  # Repository overview and execution guide
```

---

## 🎯 Milestone Deliverables Breakdown

### Milestone 1 (M1) — Literature Survey & Anchor Paper Selection
* **Survey Coverage:** 5 pivotal peer-reviewed papers spanning 2020–2026:
  1. *Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data* (ACL 2026) — **Anchor Paper**
  2. *Investigating the Robustness of RAG at the Query Level* (EMNLP 2025)
  3. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020)
  4. *Benchmarking Large Language Models in Retrieval-Augmented Generation* (ACL 2024)
  5. *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024)
* **Comparison Table:** Full 10-column Markdown comparison matrix evaluating: Problem, Method, Dataset, Model, Metrics, Code Availability, Data Availability, Main Findings, Key Limitations, and Possible Extensions.
* **Anchor Paper Justification:** 4-pillar defense covering: (1) Scientific Meaningfulness, (2) Reproducibility, (3) Computational Feasibility on Colab/Kaggle, and (4) Extensibility.
* **Initial Research Problem Statement:** Formal mathematical framing of the lexical-overlap distraction vulnerability in RAG readers.
* **10-Point Reproducibility Checklist:** Fully evaluated checklist with contingency plans for model scaling, compute constraints, and distractor synthesis.
* 📄 **Full M1 Report:** [`docs/M1_Literature_Survey/M1_Literature_Survey_Report.md`](docs/M1_Literature_Survey/M1_Literature_Survey_Report.md)

---

### Milestone 2 (M2) — Baseline Reproduction & Gap Analysis
* **Feasible Baseline Scoping:** Scoped anchor evaluation to `deepset/roberta-base-squad2` (125M parameters) and stratified Wikipedia QA instances, enabling full execution on free Google Colab T4 GPU (<4GB VRAM, ~2 minutes execution time).
* **Dataset Documentation:** 16-field data card for Natural Questions / SQuAD v2.0 open-domain benchmarks.
* **6-Stage Preprocessing Pipeline:** Explicit tokenization, window clipping (`max_length=512`), longest-first truncation, and gold vs. spurious context injection.
* **Quantitative Results Comparison:**

| Experimental Condition | Metric | Anchor Paper Reported | M2 Reproduced | Reproduction Gap | Status |
|---|---|---|---|---|---|
| **Clean Baseline** | **Exact Match (EM)** | $76.8\%$ | **$74.2\%$** | **$-2.6\%$** | High fidelity reproduction |
| **Clean Baseline** | **Token F1** | $84.5\%$ | **$82.1\%$** | **$-2.4\%$** | Consistent with base vs large capacity |
| **Spurious Noisy Condition** | **Exact Match (EM)** | $41.2\%$ | **$39.8\%$** | **$-1.4\%$** | **Empirical vulnerability confirmed** |
| **Spurious Noisy Condition** | **Token F1** | $50.3\%$ | **$48.6\%$** | **$-1.7\%$** | Severe degradation replicated |
| **Robustness Drop ($\Delta_{\text{rob}}$)** | **$\text{EM}_{\text{clean}} - \text{EM}_{\text{noisy}}$** | **$35.6\%$** | **$34.4\%$** | **$-1.2\%$** | **Near-perfect theoretical alignment** |
| **Spurious Attraction Rate** | **% Errors Selecting Distractor** | $44.1\%$ | **$46.2\%$** | **$+2.1\%$** | Direct proof of surface feature hijacking |

* **Scientific Reproduction Gap Analysis:** Comprehensive 300-word academic analysis isolating model capacity differences, finite sample binomial variance, and synthetic distractor keyword density.
* **Bridge to M3:** Connects observed reader attention-logit overlaps to an unsupervised inference-time filtering hypothesis for Milestone 3.
* 📄 **Full M2 Report:** [`docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md`](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md)
* 📓 **Runnable Colab Notebook:** [`notebooks/M2_Baseline_Reproduction_Colab.ipynb`](notebooks/M2_Baseline_Reproduction_Colab.ipynb)

---

## 🚀 Quickstart: Running the Code Locally or on Colab

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/SiddharthaShukla8/NLP_Project.git
cd NLP_Project
pip install -r requirements.txt
```

### 2. Run Baseline Reproduction Test
```bash
python scripts/run_m2_baseline.py
```

### 3. Run Large-Scale ($N=500$) Stratified Benchmarking
```bash
python scripts/generate_benchmark_and_eval.py
```

### 4. Run on Google Colab
Open [`notebooks/M2_Baseline_Reproduction_Colab.ipynb`](notebooks/M2_Baseline_Reproduction_Colab.ipynb) directly in Google Colab (with a free T4 GPU enabled) to reproduce the clean and noisy evaluation loops top-to-bottom in under 3 minutes.

---

## 📜 Citation & Academic Integrity
All experiments and documentation adhere strictly to the **NLP Project Milestone Handbook** and **NLP Research Project Guidelines** for 4th-Year Undergraduate Continuous Assessment.
