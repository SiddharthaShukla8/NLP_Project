# M4 — Controlled Experiments & Ablation Study
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## 1. Experimental Setup & Configuration Tables

To evaluate our pre-registered hypothesis from Milestone 3, we execute a tri-condition controlled experiment followed by a 5-point parameter ablation study using real, unsimulated machine learning inference on the validation split of SQuAD v2.0.

### 1.1 Baseline Configuration Table

| Parameter / Dimension | Configuration Setting | Source / Operational Rationale |
|---|---|---|
| **Model** | `deepset/roberta-base-squad2` | 124M parameter transformer reader checkpoint |
| **Dataset** | `rajpurkar/squad_v2` | Validation answerable subset from Hugging Face Hub |
| **N** | 50 instances | Held-out validation sample evaluated deterministically |
| **Device** | Apple Silicon MPS / Google Colab T4 | Academic standard compute environment |
| **Seed** | 42 | Fixed random seed for dataset iteration reproducibility |
| **Max Length** | 512 tokens | Model max sequence limit with `only_second` truncation |
| **Metric Library** | `evaluate.load('squad')` | Official Hugging Face SQuAD v2 evaluation script |
| **Threshold** | N/A for baseline | Baseline conditions evaluate raw/unfiltered text |

### 1.2 Experimental Configuration Table

| Parameter / Dimension | Configuration Setting | Mechanism / Role |
|---|---|---|
| **Condition A (Clean Baseline)** | Gold context only | Benchmark upper bound ceiling |
| **Condition B (Noisy Unfiltered)** | Spurious distractor prepended | Simulated retrieval noise with query keyword overlap |
| **Condition C (Noisy Filtered)** | Sentence-level token overlap filter ($\tau = 0.3$) | Excises sentences exceeding keyword overlap threshold |
| **Scoring Formula** | $\text{Overlap}(s, q) = \frac{|\mathcal{T}_q \cap \mathcal{T}_s|}{|\mathcal{T}_q|}$ | Non-stopword directional query token overlap ratio |
| **Ablation Range** | $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ | Threshold parameter sensitivity sweep |

---

## 2. Experimental Results: Controlled Tri-Condition Comparison

The table below presents the **actual, unsimulated quantitative results** computed via `scripts/run_m4_experiments.py` and logged in `results/m4_experiment_results.json`:

| Condition | EM | F1 | vs Clean Baseline |
|---|:---:|:---:|:---:|
| **Condition A: Clean Baseline** | **68.00%** | **73.17%** | — (Upper Bound Ceiling) |
| **Condition B: Noisy Unfiltered** | **54.00%** | **60.98%** | **-14.00% EM** (Robustness Drop) |
| **Condition C: Noisy Filtered (threshold 0.3)** | **16.00%** | **16.14%** | **-52.00% EM** (-38.00% vs. Noisy) |

---

## 3. Predicted vs. Actual Comparison

* **Predicted:** 8-12% improvement in noisy EM (bringing noisy EM from 54% toward clean EM of 68%)
* **Actual:** 38% degradation (EM dropped from 54% to 16%)
* **Verdict:** Hypothesis REFUTED

---

## 4. Ablation Study: Overlap Threshold Sensitivity

To assess whether the severe performance collapse was specific to threshold $\tau = 0.3$ or an intrinsic property of the filtering heuristic, we systematically swept across $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$. The table records the Exact Match, Token F1, and average sentences removed across all 50 evaluation instances:

| Threshold | EM | F1 | Sentences Removed (avg) |
|:---:|:---:|:---:|:---:|
| **0.1** | **14.00%** | **14.14%** | 3.12 |
| **0.2** | **12.00%** | **12.14%** | 2.84 |
| **0.3** | **16.00%** | **16.14%** | 2.42 |
| **0.4** | **20.00%** | **21.48%** | 2.10 |
| **0.5** | **30.00%** | **32.78%** | 1.60 |
