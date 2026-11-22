# M3 — Research Gap, Formal Hypothesis, and Experimental Plan
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## Section 1 — Research Gap (200 words)

The SURE framework proposed by Yang et al. (ACL 2026) rigorously formalizes how implicit spurious features (style, source, logic, format, and metadata artifacts) in retrieved grounding data induce severe performance collapses in Retrieval-Augmented Language Models (RALMs). However, existing mitigation paradigms in the surveyed literature—including the anchor paper's synthetic data augmentation and contrastive alignment—focus predominantly on model fine-tuning or end-to-end retraining. These approaches require massive compute clusters, high-end GPU memory, and access to model parameter weights, rendering them prohibitive for enterprise deployments utilizing frozen black-box models or resource-constrained edge environments.

Crucially, no prior work in the surveyed literature investigates whether a lightweight, zero-shot, inference-time preprocessing filter—operating strictly as an input-level transformation without model modifications or parameter updates—can detect and neutralize spurious grounding features before they reach the reader module. Addressing this gap is computationally significant: if an unsupervised sentence-level filtering heuristic can reliably identify query-correlated distractor carriers and prune them prior to self-attention encoding, it would provide an immediate, compute-free defense layer for vulnerable RAG pipelines.

---

## Section 2 — One Formal Hypothesis

In strict accordance with the required academic format (*"Changing X will improve/decrease Y because Z. We predict approximately ___ improvement/change"*):

**"Changing the input grounding context via an unsupervised token-overlap filter—which scores each sentence by its lexical overlap ratio with non-stopword question keywords and excises sentences exceeding a predefined threshold $\tau$ prior to reader ingestion—will improve the Exact Match (EM) score under the spurious noisy grounding condition because removing high-overlap distractor sentences prevents the reader model's cross-attention heads from being hijacked by deceptive non-causal carriers, thereby restoring attention focus to the genuine gold answer span. We predict approximately an 8–12% EM improvement under the noisy condition (elevating noisy EM from 54.00% toward 62.00%–66.00%)."**

- **Independent Variable ($X$):** Application of sentence-level token-overlap filtering on the grounding context prior to reader model inference.
- **Dependent Variable ($Y$):** Exact Match (EM) score under the spurious noisy grounding condition.
- **Mechanism ($Z$):** Pruning high-overlap distractor sentences before transformer forward passes prevents attention dispersion across deceptive non-causal text.
- **Predicted Effect:** Approximately $+8\%$ to $+12\%$ absolute recovery in Exact Match (EM).

---

## Section 3 — Prediction Before Experiment

> **Official Pre-Registration Record:**  
> This prediction is recorded formally **before running any M4 filtering experiment**.  
> - Baseline Clean EM (from M2): **68.00%**  
> - Baseline Noisy Unfiltered EM (from M2): **54.00%** (Robustness Drop = **14.00%**)  
> - **Expected Noisy Filtered EM (Condition C): approximately 62.00% – 66.00%** (an expected recovery of $+8.00\%$ to $+12.00\%$ EM).

---

## Section 4 — Experimental Plan

Our controlled empirical investigation proceeds through the following numbered protocol:

1. **Sentence-Level Segmentation & Tokenization:**  
   Segment the retrieved multi-sentence grounding context into discrete sentences using punctuation-boundary regex (`(?<=[.!?])\s+`). For both the user query $q$ and each context sentence $s$, extract content words after filtering out standard closed-class English stopwords:
   $$\mathcal{T}_q = \{w \in \text{tokens}(q) \mid w \notin \text{Stopwords}\}, \quad \mathcal{T}_s = \{w \in \text{tokens}(s) \mid w \notin \text{Stopwords}\}$$

2. **Lexical Overlap Scorer Formulation:**  
   Compute the directional keyword overlap ratio between the question and sentence:
   $$\text{Overlap}(s, q) = \frac{|\mathcal{T}_q \cap \mathcal{T}_s|}{|\mathcal{T}_q|}$$

3. **Filtering Mechanism:**  
   Set an initial default threshold $\tau = 0.3$. Retain only sentences where $\text{Overlap}(s, q) \le \tau$. If all sentences are flagged, fall back safely to the original context to avoid empty reader inputs.

4. **Tri-Condition Controlled Evaluation:**  
   Execute inference across all $N=50$ held-out answerable instances under three strictly controlled conditions:
   - **Condition A (Clean Baseline):** Model receives clean gold Wikipedia passage only.
   - **Condition B (Noisy Unfiltered):** Model receives gold passage prepended with the spurious lexical distractor.
   - **Condition C (Noisy Filtered):** Model receives noisy context preprocessed through the token-overlap filter.

5. **Standard Metric Computation:**  
   Compute Exact Match (EM) and Token F1 for all three conditions using the official Hugging Face `evaluate.load('squad')` evaluation metric.

6. **Threshold Sensitivity Ablation Study:**  
   Systematically sweep the threshold parameter $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ on Condition C to construct the sensitivity curve and identify the optimal trade-off between distractor removal and accidental gold passage excision.

7. **Predicted vs. Actual Verification:**  
   Quantitatively compare the observed recovery metric $\Delta_{\text{recovery}} = \text{EM}_{\text{filtered}} - \text{EM}_{\text{unfiltered}}$ against our pre-registered prediction of $+8.00\%$ to $+12.00\%$.
