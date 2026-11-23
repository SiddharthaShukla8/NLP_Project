<div align="center">

# 🔬 RAG Robustness & Reliability
### Quantifying Spurious Grounding Vulnerability in Retrieval-Augmented Language Models
### and Evaluating Inference-Time Lexical Filtering as a Mitigation Strategy

**NLP Research Project — Research Track | 4th Year Undergraduate**  
**B.Tech Computer Science & Artificial Intelligence**  
**Rishihood University (Newton School of Technology)**

---

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers_5.21-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)
![Commits](https://img.shields.io/badge/Commits-35+-brightgreen?style=flat-square)

</div>

---

## 📌 Project Overview

This project is a full end-to-end NLP research investigation conducted across **6 milestones**, following the standard academic research cycle:

> **Literature Survey → Baseline Reproduction → Hypothesis Formation → Controlled Experiments → Failure Analysis → IEEE Paper**

We investigate a critical vulnerability in modern **Retrieval-Augmented Generation (RAG)** systems: when retrieved documents contain *spurious features* — non-causal surface patterns that share lexical overlap with user queries — even state-of-the-art language models are significantly distracted, leading to measurable accuracy degradation.

Our anchor paper (Yang et al., ACL 2026) introduced the **SURE framework** to formally quantify this vulnerability. We reproduce the core finding, then design and test our own hypothesis: *can a zero-shot, inference-time token-overlap filter recover the accuracy loss without any model retraining?*

**The answer, rigorously demonstrated: No.** The filter caused catastrophic degradation due to collateral deletion of gold evidence — a scientifically meaningful negative result that reveals a fundamental limitation of lexical overlap as a spurious detection signal.

---

## 📄 Anchor Paper

| Field | Detail |
|-------|--------|
| **Title** | Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data |
| **Authors** | Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong et al. |
| **Venue** | ACL 2026 |
| **arXiv** | [arXiv:2503.05587](https://arxiv.org/abs/2503.05587) |
| **Framework** | SURE (Spurious Feature Robustness Evaluation) |
| **Benchmark** | SIG (Spurious Injected Grounding) |

---

## 👥 Team

| Name | Enrollment No. | Role |
|------|---------------|------|
| **Siddhartha Shukla** | 230104 | Lead Researcher & Implementation |
| **Sugat Athawale** | 230109 | Literature Survey & Analysis |
| **Ayush Dev** | 230061 | Experimentation & Ablation |
| **Piyush Kaushal** | 230112 | Report Writing & Documentation |

---

## 🎯 Research Journey (6 Milestones)

### M1 — Literature Survey
Surveyed 5 papers in the RAG robustness space. Built a 10-column comparison matrix. Selected anchor paper with a rigorous 4-pillar justification (scientifically meaningful, reproducible, feasible, extensible). Completed a 10-point reproducibility checklist.

**Papers surveyed:** Lewis et al. NeurIPS 2020 (foundational RAG) · Perçin et al. ACL 2025 (query robustness) · Lyu et al. ACL 2024 (noise benchmarking) · Liu et al. TACL 2024 (context positioning) · Yang et al. ACL 2026 (anchor)

### M2 — Baseline Reproduction
Reproduced the core finding from the anchor paper using `deepset/roberta-base-squad2` (124M parameters) and `rajpurkar/squad_v2` (N=50). Real model inference, real HuggingFace datasets, real `evaluate.load('squad')` metrics — no simulation.

**Key result:** Clean EM dropped from **68.00%** to **54.00%** under spurious distractor injection — a **14% robustness drop** confirming the anchor paper's central thesis.

### M3 — Research Gap & Hypothesis
Identified that the SURE framework only proposes fine-tuning-based mitigation. No prior work studied zero-shot inference-time filtering. Formulated a formal hypothesis before running any experiment:

> *"Applying a token-overlap Jaccard filter to grounding context will improve Exact Match under spurious noisy grounding because removing high-overlap distractor sentences before inference prevents the reader's attention from being captured by query-correlated spurious content. We predict approximately 8–12% EM improvement, bringing noisy EM from 54% toward clean EM of 68%."*

### M4 — Controlled Experiments & Ablation
Implemented the Jaccard token-overlap filter. Ran 3 conditions on all 50 examples. Swept thresholds [0.1, 0.2, 0.3, 0.4, 0.5].

**Result: Hypothesis REFUTED.** The filter caused EM to drop from 54% to 16% — worse than the unfiltered noisy condition. Root cause: 44% collateral deletion of gold evidence sentences (gold Wikipedia passages share vocabulary with questions by design).

### M5 — Failure Analysis
Analysed all 50 prediction transitions using a 2×2 outcome matrix. Filter hurt 22 cases, helped only 3. Documented 5 representative failure cases from real experiment JSON. Identified the fundamental limitation: Jaccard overlap cannot distinguish spurious distractors from causal evidence when both share query vocabulary.

### M6 — IEEE Research Paper
Wrote a complete 14-section IEEE-style research paper covering the full research arc. Explicitly separated reproduced work from original student contributions. All results sections contain only real computed numbers.

---

## 📊 Key Results at a Glance

| Condition | Exact Match | Token F1 | vs Clean Baseline |
|-----------|:-----------:|:--------:|:-----------------:|
| ✅ Clean Baseline | **68.00%** | **73.17%** | — |
| ⚠️ Spurious Noisy (unfiltered) | **54.00%** | **60.98%** | −14.00% EM |
| ❌ Token-Overlap Filtered | **16.00%** | **16.14%** | −52.00% EM |

**Ablation Study (Threshold vs EM):**

| Threshold | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| EM (%) | 14.0 | 12.0 | 16.0 | 20.0 | **30.0** |

Best threshold (0.5) still achieves only 30% EM — far below the 54% noisy baseline.

**M5 Outcome Matrix (N=50):**

| | Filtered ✅ | Filtered ❌ |
|---|:---:|:---:|
| **Noisy ✅** | 5 (no change needed) | 22 (filter hurt) |
| **Noisy ❌** | 3 (filter helped) | 20 (both wrong) |

---

## 🗂️ Milestone Reports

| Milestone | Markdown | PDF | Summary |
|-----------|----------|-----|---------|
| M1 Literature Survey | [.md](docs/M1_Literature_Survey/M1_Literature_Survey_Report.md) | [.pdf](docs/M1_Literature_Survey/M1_Literature_Survey_Report.pdf) | 5 papers, comparison table, checklist |
| M2 Baseline Reproduction | [.md](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md) | [.pdf](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.pdf) | 14% drop confirmed, real inference |
| M3 Research Gap & Hypothesis | [.md](docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.md) | [.pdf](docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.pdf) | Formal hypothesis, pre-experiment prediction |
| M4 Controlled Experiments | [.md](docs/M4_Controlled_Experiments/M4_Controlled_Experiments_Report.md) | [.pdf](docs/M4_Controlled_Experiments/M4_Controlled_Experiments_Report.pdf) | Hypothesis refuted, ablation, negative result |
| M5 Failure Analysis | [.md](docs/M5_Failure_Analysis/M5_Failure_Analysis_Report.md) | [.pdf](docs/M5_Failure_Analysis/M5_Failure_Analysis_Report.pdf) | 2×2 matrix, 5 failure cases, limitations |
| M6 IEEE Paper (Final Report) | [.md](docs/M6_Final_Paper/M6_Final_IEEE_Paper.md) | [Final Report (PDF)](docs/M6_Final_Paper/Final%20Report%20in%20IEEE.pdf) | Complete IEEE conference research paper |

---

## 📁 Repository Structure

```
NLP_Project/
│
├── 📂 docs/                          # All milestone reports (Markdown + PDF)
│   ├── M1_Literature_Survey/
│   ├── M2_Baseline_Reproduction/
│   ├── M3_Research_Gap_Hypothesis/
│   ├── M4_Controlled_Experiments/
│   ├── M5_Failure_Analysis/
│   └── M6_Final_Paper/               # Final IEEE Report (PDF + Markdown)
│
├── 📂 notebooks/
│   └── M2_Baseline_Reproduction_Colab.ipynb   # Runnable on Google Colab T4 GPU
│
├── 📂 scripts/
│   ├── run_m2_baseline.py            # M2: Real baseline inference
│   ├── run_m4_experiments.py         # M4: Filter experiment + ablation
│   └── convert_docs_to_pdf.py        # Converts all .md reports to PDF
│
├── 📂 results/
│   ├── real_m2_baseline_results.json # M2 genuine model outputs
│   └── m4_experiment_results.json    # M4 full experiment + ablation data
│
├── 📂 report_assets/
│   ├── generate_diagrams.py          # Generates all 5 research diagrams
│   ├── main.tex                      # IEEE LaTeX paper source
│   ├── fig1_rag_pipeline.png
│   ├── fig2_m2_results.png
│   ├── fig3_m4_conditions.png
│   ├── fig4_ablation.png
│   └── fig5_outcome_matrix.png
│
├── 📂 NLP_Project_Report/            # Submission directory with Final Report PDF & LaTeX
├── 📄 requirements.txt
└── 📄 README.md
```

---

## ⚙️ Setup & Reproduction

### Prerequisites
```bash
git clone https://github.com/SiddharthaShukla8/NLP_Project.git
cd NLP_Project
pip install -r requirements.txt
```

### Run M2 Baseline (Real Model Inference)
```bash
python scripts/run_m2_baseline.py
# Output: results/real_m2_baseline_results.json
# Expected: Clean EM = 68.00%, Noisy EM = 54.00%
```

### Run M4 Filter Experiment + Ablation
```bash
python scripts/run_m4_experiments.py
# Output: results/m4_experiment_results.json
# Expected: Filtered EM = 16.00% (hypothesis refuted)
```

### Run in Google Colab
Open `notebooks/M2_Baseline_Reproduction_Colab.ipynb` in Google Colab.  
Go to **Runtime → Change runtime type → GPU (T4)** before running.

### Generate Research Diagrams
```bash
python report_assets/generate_diagrams.py
# Generates: fig1–fig5 PNG files at 300 DPI
```

### Compile LaTeX Paper
```bash
cd NLP_Project_Report
pdflatex main.tex
pdflatex main.tex       # Run twice for references
```

---

## 🧪 Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | 3.13 |
| Deep Learning | PyTorch | 2.15 |
| NLP Models | HuggingFace Transformers | 5.21 |
| Datasets | HuggingFace Datasets | 5.1 |
| Evaluation | HuggingFace Evaluate | 0.4.6 |
| Model Used | deepset/roberta-base-squad2 | 124M params |
| Dataset Used | rajpurkar/squad_v2 | N=50 validation |
| Visualisation | Matplotlib + Seaborn | latest |
| Report Format | IEEE LaTeX (IEEEtran) | — |

---

## 🔑 Original Contributions (Beyond Reproduction)

1. **Feasibility reproduction** of SURE's core distraction phenomenon on a 124M-parameter open-source reader — confirming the effect without proprietary infrastructure
2. **Design and implementation** of a zero-shot Jaccard token-overlap filter as an inference-time spurious mitigation hypothesis
3. **Controlled ablation study** across 5 Jaccard thresholds — first systematic threshold sensitivity analysis for lexical spurious filtering on SQuAD v2
4. **Failure analysis** via 2×2 outcome matrix revealing collateral deletion as the primary failure mode of lexical spurious filtering
5. **Evidence** that lexical overlap is fundamentally insufficient as a spurious detection signal — motivating semantic embedding-based approaches for future work

---

## 📜 References

1. Yang et al. (2026). *Quantifying and Improving the Robustness of RALMs Against Spurious Features in Grounding Data.* ACL 2026. [arXiv:2503.05587](https://arxiv.org/abs/2503.05587)
2. Perçin et al. (2025). *Investigating the Robustness of RAG at the Query Level.* ACL GEM² 2025. [arXiv:2507.06956](https://arxiv.org/abs/2507.06956)
3. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
4. Lyu et al. (2024). *Benchmarking Large Language Models in Retrieval-Augmented Generation.* ACL 2024.
5. Liu et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024.

---

<div align="center">

**Rishihood University (Newton School of Technology) · B.Tech CS & AI · 2023–2027**  
Siddhartha Shukla · Sugat Athawale · Ayush Dev · Piyush Kaushal

</div>
