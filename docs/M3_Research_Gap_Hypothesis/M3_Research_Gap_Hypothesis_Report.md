# M3 — Research Gap, Formal Hypothesis, and Experimental Plan
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## Section 1 — Research Gap (225 words)

The SURE (Spurious feature Evaluation) framework proposed by Yang et al. (ACL 2026) rigorously formalizes how implicit spurious features—encompassing stylistic biases, authoritative source artifacts, fallacious logic shifts, format anomalies, and metadata distortions—in retrieved grounding data induce acute performance degradation across modern Retrieval-Augmented Language Models (RALMs). However, a critical limitation of the anchor paper and the surrounding literature is that the mitigation strategies proposed rely almost exclusively on compute-heavy supervised fine-tuning, synthetic adversarial data augmentation, and contrastive representation learning during training. 

In production environments, edge devices, and resource-constrained academic deployments, retraining or fine-tuning 7B-to-70B parameter reader models is practically infeasible due to prohibitive GPU memory requirements, execution latencies, and frequent lack of access to proprietary model weights. Crucially, no prior work in the surveyed literature investigates whether a lightweight, zero-shot, inference-time preprocessing filter—operating purely as a context-level sanitization heuristic without altering reader model weights—can detect and neutralize query-correlated spurious features before they reach the reader module. 

Addressing this research gap is scientifically and computationally significant: if an unsupervised sentence-level filtering mechanism can distinguish between spurious carrier sentences and causal evidentiary facts prior to cross-attention computation, it would establish an accessible, zero-cost robustness layer for arbitrary pre-trained RAG architectures.

---

## Section 2 — One Formal Hypothesis

In exact accordance with the required academic specification:

"Changing [applying a token-overlap Jaccard filter to grounding context] will improve [Exact Match under spurious noisy grounding] because [removing high-overlap distractor sentences before inference prevents the reader's attention from being captured by query-correlated spurious content]. We predict approximately 8-12% EM improvement, bringing noisy EM from 54% toward clean EM of 68%."

### Operational Variables:
- **Independent Variable ($X$):** Application of an unsupervised sentence-level token-overlap Jaccard filter to the retrieved context prior to reader model inference.
- **Dependent Variable ($Y$):** Reader model Exact Match (EM) score under spurious query-correlated noisy grounding.
- **Hypothesized Mechanism ($Z$):** Excising high-overlap non-causal sentences prevents the reader's self- and cross-attention heads from dispersing attention away from the genuine causal answer span.
- **Quantitative Target:** $+8.00\%$ to $+12.00\%$ absolute EM recovery (elevating noisy EM from $54.00\%$ toward $62.00\%\text{--}66.00\%$).

---

## Section 3 — Prediction Recorded Before Experiment

> **Official Pre-Registration Timestamp:**  
> **Status:** Recorded before experiment execution on October 29, 2026 (prior to executing M4 scripts).  
> **Evaluation Checkpoint:** `deepset/roberta-base-squad2`  
> **Evaluation Dataset:** `rajpurkar/squad_v2` ($N=50$ held-out validation instances)  
> 
> - **Condition A (Clean Gold Baseline):** $68.00\%$ Exact Match (EM)  
> - **Condition B (Noisy Unfiltered Baseline):** $54.00\%$ Exact Match (EM) (Observed Robustness Drop = $14.00\%$ EM)  
> - **Condition C (Noisy Filtered Prediction):** $62.00\%\text{--}66.00\%$ Exact Match (EM)  
> - **Predicted Recovery ($\Delta_{\text{recovery}}$):** Approximately $+8.00\%$ to $+12.00\%$ EM restoration without fine-tuning.
