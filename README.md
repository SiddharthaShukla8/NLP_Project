# RAG Robustness & Reliability
## NLP Research Project — Research Track
**B.Tech Computer Science & AI | Rishihood University (Newton School of Technology)**

> Investigating the vulnerability of Retrieval-Augmented Language Models to spurious features in grounding data, and evaluating the effectiveness of inference-time lexical filtering as a mitigation strategy.

---

## 📄 Anchor Paper
**Title:** Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data  
**Venue:** ACL 2026 | **arXiv:** [2503.05587](https://arxiv.org/abs/2503.05587)  
**Authors:** Shiping Yang et al. (SFU, Microsoft, Atlassian)

---

## 👥 Team
| Name | Enrollment |
|------|------------|
| Siddhartha Shukla | 230104 |
| Piyush Kaushal | 230112 |
| Sugat Athawale | 230109 |
| Ayush Dev | 230061 |

---

## 🔬 Key Results

| Condition | Exact Match | Token F1 |
|-----------|-------------|----------|
| Clean Baseline | 68.00% | 73.17% |
| Spurious Noisy | 54.00% | 60.98% |
| Token-Overlap Filtered | 16.00% | 16.14% |

**Finding:** A 14% robustness drop under spurious grounding is confirmed. A zero-shot Jaccard token-overlap filter was hypothesised to recover this drop but instead caused catastrophic degradation (−38% EM) due to 44% collateral deletion of gold evidence sentences — a significant negative result demonstrating that lexical overlap is insufficient as a spurious detection signal.

---

## 📁 Repository Structure

```
NLP_Project/
├── NLP_Project_Report/                    # Compilable IEEE conference report source and assets
│   ├── main.tex                           # Full IEEE LaTeX manuscript (compilable with pdflatex)
│   ├── fig1_rag_pipeline.png              # Diagram 1: RAG Architecture & Spurious Distractor Injection
│   ├── fig2_m2_results.png                # Diagram 2: M2 Baseline Clean vs. Noisy Bar Chart
│   ├── fig3_m4_conditions.png            # Diagram 3: M4 Tri-Condition Performance Comparison
│   ├── fig4_ablation.png                  # Diagram 4: Parameter Sensitivity & Ablation Sweep
│   ├── fig5_outcome_matrix.png            # Diagram 5: 2x2 Outcome Transition Heatmap
│   └── README_LaTeX.txt                   # Compilation instructions (pdflatex main.tex twice)
├── NLP_Project_Report.zip                 # Self-contained archive with LaTeX source and all 300 DPI figures
├── docs/                                  # Milestone reports in Markdown and compiled PDF formats
│   ├── M1_Literature_Survey/              # 5-paper survey, 10-column table, reproducibility checklist
│   │   ├── M1_Literature_Survey_Report.md
│   │   └── M1_Literature_Survey_Report.pdf
│   ├── M2_Baseline_Reproduction/          # Honest scoping decision matrix, data card, reproduction gap
│   │   ├── M2_Baseline_Reproduction_Report.md
│   │   └── M2_Baseline_Reproduction_Report.pdf
│   ├── M3_Research_Gap_Hypothesis/        # Research gap, formal hypothesis, pre-experiment prediction
│   │   ├── M3_Research_Gap_Hypothesis_Report.md
│   │   └── M3_Research_Gap_Hypothesis_Report.pdf
│   ├── M4_Controlled_Experiments/         # Tri-condition evaluation, 5-point ablation sweep, negative result
│   │   ├── M4_Controlled_Experiments_Report.md
│   │   └── M4_Controlled_Experiments_Report.pdf
│   ├── M5_Failure_Analysis/               # 5 case audits, 2x2 outcome matrix, 5 limitations, 3 threats
│   │   ├── M5_Failure_Analysis_Report.md
│   │   └── M5_Failure_Analysis_Report.pdf
│   ├── M6_Final_Paper/                    # Complete 14-section IEEE conference publication paper
│   │   ├── M6_Final_IEEE_Paper.md
│   │   └── M6_Final_IEEE_Paper.pdf
│   └── README.md                          # Milestone index linking all Markdown and PDF reports
├── notebooks/
│   └── M2_Baseline_Reproduction_Colab.ipynb # Executable Google Colab notebook for M2 baseline
├── report_assets/                         # High-resolution 300 DPI diagrams and figure generation scripts
│   ├── fig1_rag_pipeline.png
│   ├── fig2_m2_results.png
│   ├── fig3_m4_conditions.png
│   ├── fig4_ablation.png
│   ├── fig5_outcome_matrix.png
│   ├── generate_diagrams.py               # Standalone Python script reproducing all 5 figures
│   └── main.tex
├── results/                               # Machine-generated empirical evaluation logs
│   ├── m4_experiment_results.json         # Raw evaluation metrics and sample predictions for M4
│   └── real_m2_baseline_results.json      # Raw baseline evaluation metrics for M2
├── scripts/                               # Executable Python research scripts
│   ├── convert_docs_to_pdf.py             # Automated pipeline converting markdown reports to styled PDFs
│   ├── run_m2_baseline.py                 # Real PyTorch reader inference for M2 baseline reproduction
│   └── run_m4_experiments.py              # Real PyTorch reader inference for M4 tri-condition & ablation
├── requirements.txt                       # Pinned Python package dependencies
└── README.md                              # Main repository executive documentation
```

---

## 🗂️ Milestone Reports

| Milestone | Report | Description |
|-----------|--------|-------------|
| M1 | [Literature Survey](docs/M1_Literature_Survey/M1_Literature_Survey_Report.md) | 5-paper survey, comparison table, anchor paper justification |
| M2 | [Baseline Reproduction](docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md) | Real model inference, 14% robustness drop confirmed |
| M3 | [Research Gap & Hypothesis](docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.md) | Formal hypothesis, pre-experiment prediction |
| M4 | [Controlled Experiments](docs/M4_Controlled_Experiments/M4_Controlled_Experiments_Report.md) | Ablation study, hypothesis refuted, negative result |
| M5 | [Failure Analysis](docs/M5_Failure_Analysis/M5_Failure_Analysis_Report.md) | 2x2 outcome matrix, collateral deletion explained |
| M6 | [IEEE Paper](docs/M6_Final_Paper/M6_Final_IEEE_Paper.md) | Complete research paper, all 14 sections |

---

## ⚙️ Setup & Reproduction

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/SiddharthaShukla8/NLP_Project.git
cd NLP_Project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run M2 Baseline Reproduction
Evaluates `deepset/roberta-base-squad2` on Clean vs. Spurious Noisy grounding context across $N=50$ held-out SQuAD v2 validation instances:
```bash
python scripts/run_m2_baseline.py
```
*Outputs are saved to `results/real_m2_baseline_results.json`.*

### 3. Run M4 Controlled Experiments & Ablation Sweep
Executes the tri-condition evaluation (Clean, Noisy Unfiltered, Noisy Filtered at $\tau=0.3$) and sweeps over $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$:
```bash
python scripts/run_m4_experiments.py
```
*Outputs are saved to `results/m4_experiment_results.json`.*

### 4. Regenerate Research Diagrams & PDF Reports
```bash
# Generate high-resolution figures (300 DPI)
python report_assets/generate_diagrams.py

# Convert all markdown milestone reports to PDF
python scripts/convert_docs_to_pdf.py
```

---

## 📊 Tech Stack
* **Core Libraries:** Python 3.13, PyTorch 2.15, Transformers 5.21, Datasets 5.1, Evaluate 0.4.6, HuggingFace Hub
* **Models & Corpora:** `deepset/roberta-base-squad2`, `rajpurkar/squad_v2`
* **Visualization & Publishing:** Matplotlib 3.11, Seaborn 0.13, IEEEtran LaTeX, Pisa PDF Engine

---

## 📜 References
1. **Lewis et al. (NeurIPS 2020):** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems, 33:9459–9474.
2. **Lyu et al. (ACL 2024):** *CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models.* Proceedings of the 62nd Annual Meeting of the ACL.
3. **Perçin et al. (ACL GEM² 2025):** *Investigating the Robustness of Retrieval-Augmented Generation at the Query Level.* Proceedings of the 4th Workshop on Generation, Evaluation and Metrics (GEM²), arXiv:2507.06956.
4. **Liu et al. (TACL 2024):** *Lost in the Middle: How Language Models Use Long Contexts.* Transactions of the Association for Computational Linguistics, 12:157–173.
5. **Yang et al. (ACL 2026):** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data.* Proceedings of the 64th Annual Meeting of the ACL, arXiv:2503.05587.
