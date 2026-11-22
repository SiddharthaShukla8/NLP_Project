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

---

## Section 4 — Experimental Plan & Protocol

Our controlled empirical investigation proceeds through a structured, numbered 7-stage protocol:

1. **Sentence Boundary Segmentation & Token Preprocessing:**  
   Parse the incoming grounding context document into individual sentence units $S = \{s_1, s_2, \dots, s_m\}$ utilizing punctuation-delimited regex rules (`(?<=[.!?])\s+`). For user question $q$ and each candidate context sentence $s_i$, remove standard closed-class English stopwords and punctuation to yield content-bearing token sets:
   $$\mathcal{T}_q = \{w \in \text{lower}(\text{tokens}(q)) \mid w \notin \text{Stopwords}\}$$
   $$\mathcal{T}_{s_i} = \{w \in \text{lower}(\text{tokens}(s_i)) \mid w \notin \text{Stopwords}\}$$

2. **Jaccard / Directional Keyword Overlap Formulation:**  
   Calculate the directional keyword overlap ratio between the question vocabulary and each sentence:
   $$\text{Score}(s_i, q) = \frac{|\mathcal{T}_q \cap \mathcal{T}_{s_i}|}{|\mathcal{T}_q|}$$
   This metric directly captures how aggressively a candidate sentence mirrors query keywords.

3. **Inference-Time Sentence Pruning Mechanism:**  
   Establish a pre-registered baseline decision threshold $\tau = 0.3$. Prune any sentence $s_i$ where $\text{Score}(s_i, q) > \tau$. If all candidate sentences exceed $\tau$, implement an empty-string safety fallback that returns the unmodified input to prevent downstream runtime exceptions.

4. **Tri-Condition Controlled Evaluation Setup:**  
   Feed the $N=50$ held-out SQuAD v2 validation instances through three strictly controlled evaluation regimes:
   - **Condition A (Clean Baseline):** Reader evaluates original, unperturbed gold Wikipedia paragraphs.
   - **Condition B (Noisy Unfiltered):** Reader evaluates gold paragraphs prepended with a high-overlap synthetic distractor sentence.
   - **Condition C (Noisy Filtered, $\tau=0.3$):** Reader evaluates contexts that undergo sentence-level overlap pruning prior to reader input.

5. **Official Metric Computation:**  
   Compute Exact Match (EM) and Macro Token F1 across all conditions using Hugging Face's official SQuAD evaluation script (`evaluate.load('squad')`).

6. **Systematic Ablation Threshold Schedule:**  
   To evaluate sensitivity and identify optimal hyperparameter bounds, systematically execute an ablation sweep over 5 threshold values:
   $$\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$$
   Record Exact Match, Token F1, and the average number of pruned sentences per instance at each threshold level.

7. **Hypothesis Verification & Negative Result Protocol:**  
   Compare actual observed EM recovery $\Delta = \text{EM}_{\text{filtered}} - \text{EM}_{\text{noisy}}$ against the pre-registered prediction ($+8.00\%\text{--}+12.00\%$). If $\Delta < 0$, execute a qualitative failure case audit to inspect instances of collateral damage where genuine gold sentences were erroneously pruned.
