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
