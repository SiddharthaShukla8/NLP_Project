<div align="center">

# 🧠 Quantifying Spurious Grounding Robustness in RAG Systems

### *A Reproduction and Inference-Time Filtering Investigation*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co)
[![Model](https://img.shields.io/badge/Model-deepset%2Froberta--base--squad2-green)](https://huggingface.co/deepset/roberta-base-squad2)
[![Dataset](https://img.shields.io/badge/Dataset-SQuAD%20v2-red)](https://huggingface.co/datasets/rajpurkar/squad_v2)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/SiddharthaShukla8/NLP_Project?style=social)](https://github.com/SiddharthaShukla8/NLP_Project)

**4th-Year Undergraduate NLP Research Project (Research Track)**
*Rishihood University — Newton School of Technology*

</div>

---

## 📌 Project Overview

This project is a **complete 6-milestone NLP research investigation** anchored to the ACL 2026 paper:

> **"Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data"**
> Yang et al., arXiv:2503.05587, ACL 2026

We reproduce the core vulnerability finding — that RAG systems are susceptible to **spurious features** (non-causal, surface-level distractors) in retrieved context — and design, pre-register, and test an **inference-time token-overlap filter** as a potential zero-shot remedy. The result is a rigorous **negative finding** with deep analytical value: naive lexical filters cause catastrophic collateral damage to gold answer sentences.

### 📄 Anchor Paper

| Field | Detail |
|---|---|
| **Title** | Quantifying and Improving the Robustness of RALMs Against Spurious Features |
| **Authors** | Yang et al. |
| **Venue** | ACL 2026 |
| **arXiv** | [arXiv:2503.05587](https://arxiv.org/abs/2503.05587) |
| **Core Concept** | Spurious features in grounding context degrade RAG reader performance |
| **Benchmark** | SURE (Spurious feature Robustness Evaluation) |

---

## 👥 Team

| Name | Enrollment | Role |
|---|---|---|
| **Siddhartha Shukla** | 230104 | Project Lead, M2/M4 experiments, codebase |
| **Sugat Athawale** | 230109 | M3 hypothesis design, M5 analysis |
| **Ayush Dev** | 230061 | M1 literature survey, M6 write-up |
| **Piyush Kaushal** | 230112 | M4 ablation analysis, report LaTeX |

---

## 🗺️ Research Journey: M1 → M6

### M1 — Literature Survey
We surveyed **5 papers** spanning RAG robustness, spurious feature detection, contrastive learning defenses, and query-level robustness. The anchor paper (Yang et al., ACL 2026) was selected based on reproducibility, recency, and scientific significance. A 10-column comparison table systematically evaluates each paper against methodology, dataset, metric, model, and contribution axes.

### M2 — Baseline Reproduction
We deployed `deepset/roberta-base-squad2` (124M parameters) on **SQuAD v2.0** (N=50 answerable validation instances). Two conditions were tested:
- **Condition A (Clean):** Gold context only → **EM = 68.00%, F1 = 73.17%**
- **Condition B (Noisy):** Gold context + spurious distractor sentence injected → **EM = 54.00%, F1 = 60.98%**

This reproduces the anchor paper's core finding: **a 14% EM robustness drop** from spurious feature injection.

### M3 — Research Gap + Hypothesis
**Research Gap:** No existing work tests zero-shot, inference-time preprocessing filters for spurious feature removal in extractive RAG pipelines without fine-tuning.

**Formal Hypothesis (Pre-registered):**
> *"Filtering grounding context sentences with Jaccard token-overlap score τ ≥ 0.3 against the query will recover 8–12% EM by removing spurious distractors before inference, without fine-tuning the reader."*

**Prediction (recorded before experiment):** EM under Condition C will lie in [62%, 66%].

### M4 — Experiment Execution
We ran **5 ablation thresholds** (τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}) for the token-overlap filter and measured EM under Condition C. **The hypothesis was conclusively refuted:**

- At τ = 0.3: **EM = 16%** — catastrophic collapse (-38% vs noisy baseline)
- Best result: τ = 0.5 → **EM = 30%** — still far below clean baseline

**Root cause:** The Jaccard filter deleted gold answer sentences at a **44% collateral rate**, removing the very evidence needed to answer questions.

### M5 — Failure Analysis + Discussion
A 2×2 error transition matrix confirmed the failure mode:
- 44% of gold sentences shared keyword overlap with the query → incorrectly filtered
- Spurious sentences without keyword overlap were retained untouched

**Conclusion:** Surface lexical heuristics are intrinsically confounded — they cannot distinguish spurious correlation from genuine evidence.

### M6 — Final Report
Full IEEE-format LaTeX paper written covering all milestones, submitted as `NLP_Project_Report.zip` with 5 research diagrams (300 DPI).

---

## 📊 Key Results at a Glance

| Condition | Description | Exact Match (EM) | F1 Score |
|---|---|---|---|
| **A — Clean** | Gold context only | **68.00%** ✅ | **73.17%** |
| **B — Noisy** | + Spurious distractor injected | **54.00%** ⚠️ | **60.98%** |
| **C — Filtered** | Jaccard filter τ=0.3 applied | **16.00%** ❌ | ~21% |

> **Robustness Drop (A→B):** −14.00% EM  
> **Filter Damage (A→C):** −52.00% EM  
> **Collateral Gold Deletion Rate:** 44.0%

### Ablation Study: Jaccard Threshold vs. EM

| Threshold τ | EM (%) | Verdict |
|---|---|---|
| 0.1 | 10% | Extreme over-filtering |
| 0.2 | 14% | Severe over-filtering |
| **0.3** | **16%** | **Hypothesis threshold — worst** |
| 0.4 | 24% | Still degraded |
| 0.5 | 30% | Best, but −38% vs clean |

### M5 Outcome Matrix

| | Filter **Removes** | Filter **Keeps** |
|---|---|---|
| **Spurious sentence** | ✅ True Positive | ❌ False Negative |
| **Gold sentence** | ❌ False Positive (44%) | ✅ True Negative |

---

## 📁 Repository Structure

```
NLP_Project/
│
├── 📂 docs/                                   # All milestone reports
│   ├── M1_Literature_Survey/
│   │   ├── M1_Literature_Survey_Report.md
│   │   └── M1_Literature_Survey_Report.pdf
│   ├── M2_Baseline_Reproduction/
│   │   ├── M2_Baseline_Reproduction_Report.md
│   │   └── M2_Baseline_Reproduction_Report.pdf
│   ├── M3_Research_Gap_Hypothesis/
│   │   ├── M3_Research_Gap_Hypothesis_Report.md
│   │   └── M3_Research_Gap_Hypothesis_Report.pdf
│   ├── M4_Experiment_Results/
│   │   ├── M4_Experiment_Results_Report.md
│   │   └── M4_Experiment_Results_Report.pdf
│   ├── M5_Analysis_Discussion/
│   │   ├── M5_Analysis_Discussion_Report.md
│   │   └── M5_Analysis_Discussion_Report.pdf
│   └── M6_Final_Report/
│       ├── M6_Final_Report.md
│       └── M6_Final_Report.pdf
│
├── 📂 scripts/                                # Runnable experiment scripts
│   ├── run_m2_baseline.py                     # M2: baseline reproduction
│   └── run_m4_experiments.py                  # M4: filter ablation study
│
├── 📂 notebooks/
│   └── M2_Baseline_Reproduction_Colab.ipynb   # Colab-ready notebook
│
├── 📂 results/                                # Real computed results (JSON)
│   ├── real_m2_baseline_results.json          # M2 output: EM 68%/54%
│   └── m4_experiment_results.json             # M4 output: ablation data
│
├── 📂 report_assets/                          # LaTeX report materials
│   ├── main.tex                               # IEEE-format LaTeX source
│   ├── generate_diagrams.py                   # Diagram generation script
│   ├── fig1_rag_pipeline.png                  # RAG pipeline diagram
│   ├── fig2_m2_results.png                    # M2 results bar chart
│   ├── fig3_m4_conditions.png                 # M4 condition comparison
│   ├── fig4_ablation.png                      # Ablation threshold curve
│   └── fig5_outcome_matrix.png                # 2×2 error matrix
│
├── 📂 NLP_Project_Report/                     # ZIP submission contents
│   ├── main.tex
│   └── [all 5 figures]
│
├── 📦 NLP_Project_Report.zip                  # Final submission ZIP
├── 📄 requirements.txt                        # Python dependencies
└── 📄 README.md                               # This file
```

---

## 📑 Milestone Reports

| Milestone | Markdown | PDF |
|---|---|---|
| M1 — Literature Survey | [M1_Report.md](docs/M1_Literature_Survey/M1_Literature_Survey_Report.md) | [M1_Report.pdf](docs/M1_Literature_Survey/M1_Literature_Survey_Report.pdf) |
| M2 — Baseline Reproduction | [M2_Report.md](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md) | [M2_Report.pdf](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.pdf) |
| M3 — Research Gap + Hypothesis | [M3_Report.md](docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.md) | [M3_Report.pdf](docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.pdf) |
| M4 — Experiment Results | [M4_Report.md](docs/M4_Experiment_Results/M4_Experiment_Results_Report.md) | [M4_Report.pdf](docs/M4_Experiment_Results/M4_Experiment_Results_Report.pdf) |
| M5 — Analysis + Discussion | [M5_Report.md](docs/M5_Analysis_Discussion/M5_Analysis_Discussion_Report.md) | [M5_Report.pdf](docs/M5_Analysis_Discussion/M5_Analysis_Discussion_Report.pdf) |
| M6 — Final Report | [M6_Report.md](docs/M6_Final_Report/M6_Final_Report.md) | [M6_Report.pdf](docs/M6_Final_Report/M6_Final_Report.pdf) |

---

## ⚙️ Setup & Reproduction

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/SiddharthaShukla8/NLP_Project.git
cd NLP_Project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce M2 Baseline (EM: 68% clean / 54% noisy)

```bash
python scripts/run_m2_baseline.py
# Output: results/real_m2_baseline_results.json
```

> ⚠️ First run downloads `deepset/roberta-base-squad2` (~500MB) from HuggingFace Hub.

### Reproduce M4 Filter Ablation (EM: 16% at τ=0.3)

```bash
python scripts/run_m4_experiments.py
# Output: results/m4_experiment_results.json
```

### Run on Google Colab (No local GPU needed)

Open [`notebooks/M2_Baseline_Reproduction_Colab.ipynb`](notebooks/M2_Baseline_Reproduction_Colab.ipynb) in Google Colab. All dependencies are auto-installed in the first cell.

### Regenerate Diagrams

```bash
pip install matplotlib seaborn
python report_assets/generate_diagrams.py
# Saves 5 PNG files at 300 DPI in report_assets/
```

### Compile LaTeX Report

```bash
cd NLP_Project_Report
pdflatex main.tex   # Requires a local LaTeX installation (TeX Live / MiKTeX)
```

Or open `main.tex` in [Overleaf](https://www.overleaf.com) for online compilation.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Framework** | HuggingFace Transformers 4.x |
| **Reader Model** | `deepset/roberta-base-squad2` (124M params) |
| **Dataset** | `rajpurkar/squad_v2` via HuggingFace Datasets |
| **Evaluation** | HuggingFace `evaluate` (SQuAD metric) |
| **Filtering** | Custom Jaccard token-overlap (no external deps) |
| **Visualization** | Matplotlib 3.x, Seaborn |
| **Report** | IEEE LaTeX (IEEEtran), Pandoc, xhtml2pdf |
| **Notebook** | Google Colab (`.ipynb`) |
| **Version Control** | Git + GitHub |

---

## 🔬 Original Contributions

1. **End-to-end spurious feature reproduction pipeline** — A fully self-contained Python script (`run_m2_baseline.py`) that downloads a real model and dataset, injects spurious distractors via controlled query-correlated sentence injection, and computes standardized SQuAD EM/F1 metrics.

2. **Pre-registered formal hypothesis** — Following open-science practice, the hypothesis and expected results (8–12% recovery) were formally recorded *before* any experiment was run, enabling honest evaluation of the outcome.

3. **5-point ablation study with failure taxonomy** — Systematic sweep across τ ∈ {0.1–0.5} with per-threshold EM measurement, coupled with a 2×2 error transition matrix identifying the precise failure mode (44% collateral gold deletion).

4. **Principled negative-result reporting** — Full analytical treatment of the negative finding, explaining *why* surface lexical overlap is a confounded signal, contrasting this with the semantic entailment modeling required for a valid defense.

5. **Complete reproducible research package** — All code, results (JSON), diagrams (PNG, 300 DPI), and a compiled LaTeX IEEE paper are version-controlled and publicly available, meeting full reproducibility standards.

---

## 📚 References

1. Yang et al. (2026). *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data.* ACL 2026. [arXiv:2503.05587](https://arxiv.org/abs/2503.05587)

2. Gao et al. (2025). *Investigating the Robustness of RAG at the Query Level.* arXiv:2507.06956.

3. Rajpurkar et al. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL 2018.

4. Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.

5. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.

---

<div align="center">

**Siddhartha Shukla (230104) · Sugat Athawale (230109) · Ayush Dev (230061) · Piyush Kaushal (230112)**

*B.Tech Computer Science & Artificial Intelligence — Rishihood University (Newton School of Technology)*

[📂 Repository](https://github.com/SiddharthaShukla8/NLP_Project) · [📄 LaTeX Report](NLP_Project_Report.zip) · [🤗 Model](https://huggingface.co/deepset/roberta-base-squad2) · [📊 Dataset](https://huggingface.co/datasets/rajpurkar/squad_v2)

</div>
