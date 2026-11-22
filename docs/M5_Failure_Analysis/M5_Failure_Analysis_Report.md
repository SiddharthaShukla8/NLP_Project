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
