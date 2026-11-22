# M5 — Failure Analysis, Discussion, and Limitations
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## Section 1 — Representative Failure Cases (In-Depth Audit)

To understand why the token-overlap filter failed and produced a catastrophic drop from $54.00\%$ EM down to $16.00\%$ EM in Milestone 4, we examine 5 representative failure instances extracted directly from our empirical execution logs:

---

### Failure Case 1: Filter Over-Aggression (Gold Evidence Excision)
* **Question:** *"When were the Normans in Normandy?"*
* **Gold Answer:** `['10th and 11th centuries', 'in the 10th and 11th centuries']`
* **Noisy Prediction (Before Filter):** `'10th and 11th centuries'` (Exact Match = 1.0)
* **Filtered Prediction (After Filter):** `''` (Exact Match = 0.0)
* **What Happened:** Under the unfiltered noisy condition, the reader model was robust enough to extract the correct time span from the original gold text despite the distractor. However, the filter detected heavy keyword overlap (`normans`, `normandy`) in the genuine gold sentence (*"The Normans were the people who in the 10th and 11th centuries gave their name to Normandy..."*) and pruned it. Left with no relevant evidence, the reader returned an empty string.
* **Why It Happened:** **Filter was too aggressive and removed the gold sentence.** The genuine evidentiary sentence naturally mirrors query syntax; penalizing token overlap directly amputates the answer.

---

### Failure Case 2: Query Echo Distortion & Incoherent Span Boundary
* **Question:** *"In what country is Normandy located?"*
* **Gold Answer:** `['France']`
* **Noisy Prediction (Before Filter):** `'France'` (Exact Match = 1.0)
* **Filtered Prediction (After Filter):** `'In what country is Normandy located?They were descended from Norse...'` (Exact Match = 0.0)
* **What Happened:** The filter fragmented the paragraph boundary, leaving a dangling lead clause. When the reader processed the broken context, its start and end token logit distributions became severely destabilized, spanning across query tokens rather than targeting the answer entity.
* **Why It Happened:** **Distractor removal disrupted sentence boundary coherence**, destroying the local syntactic context that the transformer self-attention heads rely on for span boundary calibration.

---

### Failure Case 3: Both Wrong (Intrinsic Model Attention Hijacking)
* **Question:** *"From which countries did the Norse originate?"*
* **Gold Answer:** `['Denmark, Iceland and Norway']`
* **Noisy Prediction (Before Filter):** `''` (Exact Match = 0.0)
* **Filtered Prediction (After Filter):** `''` (Exact Match = 0.0)
* **What Happened:** In the unfiltered condition, the distractor overwhelmed the cross-attention heads, suppressing the logit scores for the true entities. After filtering, the gold sentence (*"...raiders and pirates from Denmark, Iceland and Norway..."*) was also pruned due to matching `countries`, `norse`, and `originate`. Both conditions failed completely.
* **Why It Happened:** **Dual-mode failure.** In the unfiltered setting, the distractor captured attention; in the filtered setting, the filter excised the evidence.

---

### Failure Case 4: Pre-Existing Extraction Difficulty
* **Question:** *"Who was the Norse leader?"*
* **Gold Answer:** `['Rollo']`
* **Noisy Prediction (Before Filter):** `''` (Exact Match = 0.0)
* **Filtered Prediction (After Filter):** `''` (Exact Match = 0.0)
* **What Happened:** The clean baseline model already struggled on this specific grammatical construction (*"...who, under their leader Rollo, agreed..."*). Filtering neither helped nor harmed; the instance remained unanswerable for `roberta-base`.
* **Why It Happened:** **Parametric & architectural capacity limitation.** The model failed due to complex syntactic apposition in the source text, independent of distractor presence.

---

### Failure Case 5: Collateral Damage on Resilient Baseline Instances
* **Question:** *"What century did the Normans first gain their separate identity?"*
* **Gold Answer:** `['10th century', 'the first half of the 10th century', '10th']`
* **Noisy Prediction (Before Filter):** `'10th'` (Exact Match = 1.0)
* **Filtered Prediction (After Filter):** `''` (Exact Match = 0.0)
* **What Happened:** The gold passage (*"The Norman dynasty had a major political, cultural and military impact on medieval Europe and as far afield as the Near East. The distinct identity of the Normans emerged initially in the first half of the 10th century..."*) had substantial overlap with the query. The filter stripped this entire clause, converting a previously correct answer into a total miss.
* **Why It Happened:** **Overlap threshold too aggressive to allow causal evidence to survive.** Sentences that answer *"What century did the Normans first gain..."* inevitably repeat those exact words.

---

## Section 2 — Confusion Matrix Style Outcome Analysis

Tracking the transition of each of the $N=50$ instances between the **Noisy Unfiltered Condition (B)** and the **Noisy Filtered Condition (C)** yields the following $2 \times 2$ transition matrix:

| Outcome Category | Instance Count ($N=50$) | Percentage | Empirical Meaning |
|---|:---:|:---:|---|
| **Filter Helped** ($B_{\text{wrong}} \rightarrow C_{\text{correct}}$) | **3** | **6.0%** | Filter pruned distractor and preserved gold evidence |
| **Filter Made No Difference** ($B_{\text{wrong}} \rightarrow C_{\text{wrong}}$) | **20** | **40.0%** | Both conditions failed (hard examples or excised gold) |
| **Filter Made No Difference** ($B_{\text{correct}} \rightarrow C_{\text{correct}}$) | **5** | **10.0%** | Model remained robust across both conditions |
| **Filter Hurt** ($B_{\text{correct}} \rightarrow C_{\text{wrong}}$) | **22** | **44.0%** | Filter caused collateral damage by deleting the gold answer sentence |
| **Clean Baseline & Filtered Both Correct** | **8** | **16.0%** | Overall surviving accuracy ceiling under filtering |

### Key Analytical Finding
The filter **hurt 22 instances (44.0%)** while **helping only 3 instances (6.0%)**. This quantitative breakdown rigorously accounts for the **-38.00% net drop in Exact Match**. The heuristic suffered from a high false-positive rate: sentences containing causal answers are lexically similar to the prompt by definition, making lexical overlap an inherently flawed discriminative signal for context pruning.

---

## Section 3 — Limitations

1. **Constrained Sample Size ($N=50$):**  
   The evaluation was scoped to $N=50$ held-out answerable instances to ensure deterministic, reproducible execution on academic hardware. While sufficient for revealing macroscopic behavioral collapses, $N=50$ yields wide confidence intervals and is indicative rather than statistically significant at conventional academic thresholds ($p < 0.05$).
2. **Single-Dimension Spurious Feature Representation:**  
   The anchor paper's SURE framework evaluates five distinct, highly nuanced spurious dimensions (Style, Source, Logic, Format, and Metadata). Our experiment evaluated a single synthetic high-lexical-overlap carrier, which does not capture subtler non-lexical shortcuts such as Markdown headers or temporal metadata.
3. **Unsupervised Rule-Based Heuristic Limitations:**  
   The token-overlap filter operates strictly on surface-level bag-of-words token sets without semantic embeddings or Natural Language Inference (NLI) entailment scoring. Consequently, it cannot differentiate between *lexical overlap carrying semantic entailment* and *lexical overlap acting as a superficial distractor*.
4. **Architecture Discrepancy (Extractive vs. Generative):**  
   Our baseline utilizes `deepset/roberta-base-squad2` (124M parameter extractive encoder). While extractive readers clearly demonstrate the attention-hijacking phenomenon, modern production RAG pipelines deploy generative autoregressive LLMs (e.g., Llama-3, Mistral, GPT-4), which may exhibit different sensitivity profiles.
5. **No Separate Validation Split for Threshold Tuning:**  
   The ablation study evaluated thresholds on the same test set without a dedicated held-out dev split, presenting a risk of post-hoc overfitting had any threshold succeeded.

---

## Section 4 — Threats to Validity

1. **Internal Validity (Spurious Injection Fidelity):**  
   Our spurious passage injection used a deterministic carrier template prepended to the gold context. In live enterprise RAG systems, spurious information may be embedded naturally within multi-paragraph web snippets or distributed across multiple retrieved documents, potentially altering the distraction dynamics.
2. **External Validity (Corpus Domain Representation):**  
   Evaluating on SQuAD v2.0 Wikipedia paragraphs represents clean, encyclopedic English text with well-defined entity spans. Real-world enterprise RAG corpora (customer support transcripts, legal filings, medical notes) feature messy formatting and conflicting viewpoints where both noise and evidence distributions differ markedly.
3. **Construct Validity (Metric Stringency):**  
   Exact Match (EM) and Token F1 enforce strict character and token overlap against human reference strings. In generative RAG settings, models often generate valid semantic paraphrases that receive an EM score of zero despite being factually truthful, potentially underestimating true system utility.
