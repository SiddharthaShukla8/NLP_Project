# M5 — Failure Analysis, Limitations, and Threats to Validity
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## 1. Qualitative Failure Case Audits (Real Instances from M4 JSON)

To understand the exact failure mechanisms behind the $-38.00\%$ EM degradation observed in Milestone 4, we examine five representative instances extracted directly from `results/m4_experiment_results.json`:

### Case 1: Collateral Deletion of Causal Evidence (Filter Hurt)
- **Question Text:** `"When were the Normans in Normandy?"` (Sample #2)
- **Gold Answer:** `['10th and 11th centuries']`
- **Clean Prediction:** `"10th and 11th centuries"` (EM: 1.0, F1: 1.0)
- **Noisy Prediction:** `"10th and 11th centuries"` (EM: 1.0, F1: 1.0)
- **Filtered Prediction:** `""` (EM: 0.0, F1: 0.0)
- **What Happened:** Filter hurt
- **Why:** Collateral deletion. The gold context sentence (*"The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France."*) shared both `normans` and `normandy` with the query. Its keyword overlap score was $0.50$, exceeding the $\tau=0.30$ threshold. The filter excised the authentic gold evidence sentence, leaving the model with an amputated context and forcing an empty output.

### Case 2: Successful Pruning of Spurious Distractor (Filter Helped)
- **Question Text:** `"What river originally bounded the Duchy"` (Sample #13)
- **Gold Answer:** `['Seine', 'Epte', 'Seine']`
- **Clean Prediction:** `"Epte"` (EM: 1.0, F1: 1.0)
- **Noisy Prediction:** `""` (EM: 0.0, F1: 0.0)
- **Filtered Prediction:** `"Epte"` (EM: 1.0, F1: 1.0)
- **What Happened:** Filter helped
- **Why:** Threshold too high/low (optimal threshold balance). In the noisy condition, the synthetic carrier (*"A recent official investigation regarding river originally bounded duchy..."*) completely captured the reader's self-attention, collapsing extraction to an empty string. The filter identified the distractor's overlap ($0.75 > 0.30$) and pruned it, while the gold sentence (*"The territory of the Duchy of Normandy was bounded by the Epte river..."*) had a lower overlap score, preserving the gold span and restoring accuracy.

### Case 3: Distorted Span Extraction due to Truncation (Filter Hurt)
- **Question Text:** `"In what country is Normandy located?"` (Sample #1)
- **Gold Answer:** `['France']`
- **Clean Prediction:** `"France"` (EM: 1.0, F1: 1.0)
- **Noisy Prediction:** `"France"` (EM: 1.0, F1: 1.0)
- **Filtered Prediction:** `"In what country is Normandy located?They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia"` (EM: 0.0, F1: 0.0)
- **What Happened:** Filter hurt
- **Why:** Collateral deletion. The first sentence containing *"Normandy, a region in France"* was flagged as exceeding $\tau=0.30$ and removed. The reader was left with subsequent sentences that described historical origins rather than geographic location, causing the model to output a degenerate span.

### Case 4: Persistent Attention Disruption (Filter Made No Difference / Both Wrong)
- **Question Text:** `"From which countries did the Norse originate?"` (Sample #3)
- **Gold Answer:** `['Denmark, Iceland and Norway']`
- **Clean Prediction:** `"Denmark, Iceland and Norway"` (EM: 1.0, F1: 1.0)
- **Noisy Prediction:** `""` (EM: 0.0, F1: 0.0)
- **Filtered Prediction:** `""` (EM: 0.0, F1: 0.0)
- **What Happened:** Filter made no difference
- **Why:** Gold-distractor lexical overlap. Both the synthetic distractor and the gold answer sentence heavily overlapped with the keywords `countries`, `norse`, and `originate`. Filtering eliminated both the distractor and the gold context, leaving no valid span to extract.

### Case 5: Parametric Representation Limit (Filter Made No Difference / Both Wrong)
- **Question Text:** `"Who was Margaret's brother?"` (Sample #50)
- **Gold Answer:** `['Edgar', 'Edgar', 'Edgar Atheling']`
- **Clean Prediction:** `""` (EM: 0.0, F1: 0.0)
- **Noisy Prediction:** `""` (EM: 0.0, F1: 0.0)
- **Filtered Prediction:** `""` (EM: 0.0, F1: 0.0)
- **What Happened:** Filter made no difference
- **Why:** Model parametric bias. The model failed to predict the answer even in Condition A (Clean Context), demonstrating that failure on this instance stems from the base model's inability to resolve the complex possessive syntactic coreference (*"King Malcolm III married Edgar's sister Margaret"*), rather than retrieval noise or filtering artifacts.

---

## 2. 2x2 Outcome Transition Matrix (N=50 Instances)

To systematically assess the behavioral shifts induced by the filtering intervention across all 50 evaluation instances, we construct the 2x2 outcome matrix comparing prediction correctness under the Noisy Condition (Condition B) versus the Filtered Condition (Condition C, $\tau=0.3$):

| | Filtered Correct | Filtered Wrong | Total |
|---|:---:|:---:|:---:|
| **Noisy Correct** | **5** *(Both Correct / No Help)* | **22** *(Filter Hurt / Collateral Damage)* | **27** |
| **Noisy Wrong** | **3** *(Filter Helped / True Recovery)* | **20** *(Both Wrong / Persistent Failure)* | **23** |
| **Total** | **8** | **42** | **50** |

### Breakdown by Category:
1. **Filter Hurt (Collateral Damage): 22 instances ($44.0\%$)**  
   The model correctly answered the question in the noisy condition, but filtering degraded its performance by excising authentic gold sentences.
2. **Both Wrong / Persistent Failure: 20 instances ($40.0\%$)**  
   The model failed on the noisy input and remained incorrect after filtering.
3. **Both Correct / No Help: 5 instances ($10.0\%$)**  
   The gold context survived filtering and the model maintained correct predictions across both conditions.
4. **Filter Helped (True Recovery): 3 instances ($6.0\%$)**  
   The filter successfully eliminated the distractor without destroying the gold evidence, recovering the correct answer.

---

## 3. Key Finding from Failure Analysis

The 2x2 matrix reveals that the filter hurt more cases than it helped. The primary failure mode is collateral deletion: because gold Wikipedia passages naturally share content words with questions (they are retrieved precisely because of topical relevance), Jaccard overlap cannot reliably distinguish spurious distractors from causal evidence. This finding motivates a semantic rather than lexical approach to spurious filtering in future work.

---

## 4. Methodological Limitations (5 Distinct Constraints)

1. **Evaluation Sample Size Bound ($N=50$):**  
   The empirical evaluation is scoped to $N=50$ answerable validation instances from SQuAD v2.0 due to educational compute bounds. While this sample size reliably exposes the attention-hijacking phenomenon and the filter's failure mechanisms, it does not achieve asymptotic statistical significance at conventional academic levels ($p < 0.05$).

2. **Synthetic Distractor Uniformity:**  
   Our experimental perturbation utilized a single synthetic query-correlated assertive template rather than generating distinct datasets for all five dimensions of the SURE taxonomy (style, source, logic, format, metadata). While this design isolates direct keyword-induced attention hijacking, it does not evaluate how lexical filtering interacts with subtle stylistic or structural artifacts.

3. **Absence of Semantic Directionality in Filtering Heuristic:**  
   The token-overlap filter operates strictly on set-theoretic keyword intersection without directional semantic modeling or Natural Language Inference (NLI). It cannot determine whether a sentence logically entails an answer or merely repeats question vocabulary, treating genuine evidence and deceptive distractors identically.

4. **Extractive Encoder vs. Generative LLM Architecture:**  
   Our experiments evaluated `deepset/roberta-base-squad2`, an extractive bidirectional encoder, rather than modern autoregressive decoder LLMs (e.g., Llama-3-8B or Mistral-7B). While the underlying cross-attention vulnerability is conceptually identical, autoregressive generators may exhibit distinct hallucination behaviors when exposed to truncated or noisy contexts.

5. **In-Sample Threshold Optimization:**  
   The parameter sweep across $\tau \in [0.1, 0.5]$ was conducted on the evaluation sample without a segregated validation set for hyperparameter tuning. While sufficient for descriptive sensitivity analysis, true operational deployment would require calibrating $\tau$ on held-out development data to prevent overfitting.

---

## 5. Threats to Validity (3 Core Dimensions)

1. **Internal Validity — Tokenization and Segmentation Sensitivity:**  
   Sentence boundary segmentation relied on regex-based punctuation splits, and token matching depended on a static English stopword list. Improper sentence splits on abbreviations or numbers could distort sentence token counts, causing unintended filtering decisions that affect downstream reader performance.

2. **External Validity — Domain and Corpus Specificity:**  
   All evaluated passages originate from encyclopedic Wikipedia articles within the SQuAD v2.0 benchmark. Models and filters may demonstrate substantially different behavior on complex multi-hop corpora (HotpotQA), conversational search datasets, or highly technical domains (legal or biomedical literature) where lexical overlap patterns differ.

3. **Construct Validity — Exact Match as a Grounding Robustness Metric:**  
   Exact Match (EM) requires strict string equality with human reference annotations, scoring partial extractions or semantically equivalent paraphrases as complete failures ($0.0$). While EM provides an objective, reproducible metric for extractive QA, it may overestimate model fragility compared to soft semantic evaluation metrics or human qualitative judgment.
