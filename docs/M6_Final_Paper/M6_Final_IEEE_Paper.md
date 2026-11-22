# Quantifying and Mitigating Spurious Feature Sensitivity in Retrieval-Augmented Language Models: An Empirical Investigation

**Siddhartha Shukla**  
Department of Computer Science & Artificial Intelligence  
B.Tech 4th-Year Undergraduate Guided Research  
Repository: [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## Abstract
Retrieval-Augmented Generation (RAG) integrates non-parametric retrieval corpora with parametric neural models to reduce hallucination. However, recent benchmarks like the SURE framework reveal that Retrieval-Augmented Language Models (RALMs) are acutely vulnerable to implicit, non-causal *spurious features* in retrieved grounding data. In this study, we investigate the nature of this vulnerability and empirically evaluate an unsupervised, inference-time preprocessing intervention designed to neutralize query-correlated distractors without costly model retraining. Using an extractive reader baseline (`deepset/roberta-base-squad2`) on the SQuAD v2.0 benchmark, we first replicate the core vulnerability, observing that synthetic query-correlated distractor injection causes Exact Match (EM) to collapse from $68.00\%$ to $54.00\%$ (a $14.00\%$ robustness drop). We then evaluate a pre-registered hypothesis: that filtering sentences exhibiting high directional keyword overlap ($\tau=0.3$) prior to inference would mitigate attention hijacking and recover $8\%\text{--}12\%$ EM. Controlled experiments strongly refute this hypothesis: token-overlap filtering causes performance to deteriorate further to $16.00\%$ EM (a net recovery of $-38.00\%$). An ablation study across thresholds $\tau \in [0.1, 0.5]$ and an error transition audit reveal that naive lexical overlap acts as a double-edged sword, excising causal gold sentences at a $44.0\%$ collateral damage rate. We conclude that surface-level lexical heuristics cannot separate spurious noise from authentic evidence, establishing that robust inference-time RAG defense requires semantic entailment modeling rather than keyword pruning.

---

## 1. Introduction
Retrieval-Augmented Generation (RAG) has emerged as the standard architectural paradigm for grounding Large Language Models (LLMs) in external, factual knowledge. By prepending retrieved non-parametric context passages to user queries, RAG architectures mitigate parametric hallucinations, expand temporal coverage, and facilitate traceable attribution. Despite these advantages, modern RAG pipelines remain brittle in production environments where retrieval systems inevitably return noisy, partially relevant, or adversarial passages.

Recent research has predominantly characterized retrieval noise as *semantic irrelevance*—passages addressing unrelated topics or containing direct factual contradictions. However, the anchor paper by Yang et al. (ACL 2026) establishes that models are equally susceptible to *spurious features*: semantic-agnostic or implicit textual artifacts such as stylistic shifts, authoritative phrasing, and dense lexical matching that correlate superficially with queries without providing genuine evidentiary support. When such distractors are present, self-attention heads in reader models disproportionately attend to the distractor tokens, leading to answer hijacking.

While prior mitigation strategies focus almost entirely on compute-heavy supervised fine-tuning or contrastive pre-training, resource-constrained deployments require zero-shot, inference-time defenses. In this paper, we make three primary contributions:
1. We reproduce the empirical fragility of extractive reader models under query-correlated grounding noise, measuring a $14.00\%$ EM drop on SQuAD v2.0.
2. We design, pre-register, and execute a controlled empirical test of an unsupervised sentence-level token-overlap filter aimed at excising spurious distractors prior to reader processing.
3. We provide a rigorous failure analysis and ablation study explaining why lexical overlap heuristics fail catastrophically by penalizing authentic causal evidence.

---

## 2. Related Work
Our investigation builds upon five pivotal studies in retrieval-augmented reliability:

Foundational RAG was formalized by Lewis et al. [1], combining dense passage retrieval (DPR) with sequence-to-sequence BART generators, marginalizing over latent retrieved documents. While effective on clean corpora, foundational RAG assumes noise-free retrieval. Lyu et al. [2] introduced the CRUD-RAG benchmark, formalizing retrieval noise into irrelevant, contradictory, and counterfactual documents, reporting that counterfactual text induces $>30\%$ degradation.

Addressing input variations, Perçin et al. [3] benchmarked query-level perturbations across 1,092 experiments, showing that dense retrievers collapse under subtle query typos. Investigating positional bias, Liu et al. [4] identified the "Lost in the Middle" phenomenon, demonstrating a U-shaped performance curve where facts placed in the center of multi-document contexts suffer severe degradation. Most recently, our anchor paper by Yang et al. [5] established the SURE framework and SIG benchmark, demonstrating that state-of-the-art LLMs suffer substantial drops in Robustness Ratio when exposed to non-causal stylistic and formatting shortcuts.

---

## 3. Anchor Paper and Reproduction
The anchor paper by Yang et al. [5] (*"Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data"*, arXiv:2503.05587) defines a 5-dimension taxonomy of spurious grounding noise: Style, Source, Logic, Format, and Metadata. The authors evaluate six major LLMs (Mistral-7B, Llama-3.1-8B/70B, Qwen-2.5-7B, GPT-4o) using the SURE framework, measuring instance-level state transitions via Robustness Ratio (RR), Lose Rate (LR), and Win Rate (WR).

Operating within academic compute constraints, we scoped our baseline reproduction to `deepset/roberta-base-squad2` (124M parameters) on $N=50$ held-out answerable validation instances from `rajpurkar/squad_v2`. Evaluating on real model forward passes scored via `evaluate.load('squad')`, our clean baseline achieved an Exact Match of **68.00%** (Token F1: **73.17%**). Injecting a synthetic distractor passage carrying high query keyword overlap caused performance to collapse to **54.00% EM** (Token F1: **60.98%**), producing an authentic **14.00% robustness drop**. This validated the anchor paper's central finding: reader attention is easily seduced by query-correlated surface text.

---

## 4. Research Gap
While the SURE framework demonstrates that spurious features severely degrade RALMs, the authors propose mitigation exclusively through data synthesis and model retraining. In enterprise environments utilizing frozen commercial APIs or edge devices with strict compute budgets, model parameter modification is impossible. No prior work in the surveyed literature investigates whether an **unsupervised, zero-shot, inference-time preprocessing filter** can neutralize spurious features at the input level before they reach the reader. This represents a critical open research question for accessible, compute-free RAG robustness.

---

## 5. Proposed Hypothesis
To address this gap, we pre-registered the following formal hypothesis:

> *"Changing the input grounding context via an unsupervised token-overlap filter—which scores each sentence by its lexical overlap ratio with non-stopword question keywords and excises sentences exceeding a predefined threshold $\tau$ prior to reader ingestion—will improve the Exact Match (EM) score under the spurious noisy grounding condition because removing high-overlap distractor sentences prevents the reader model's cross-attention heads from being hijacked by deceptive non-causal carriers, thereby restoring attention focus to the genuine gold answer span. We predict approximately an 8–12% EM improvement under the noisy condition (elevating noisy EM from 54.00% toward 62.00%–66.00%)."*

---

## 6. Methodology
Our proposed inference-time intervention segments the multi-sentence retrieved context $C$ into constituent sentences $\{s_1, s_2, \dots, s_m\}$ using regex punctuation boundaries. For the query $q$ and each candidate sentence $s$, we extract filtered non-stopword token sets $\mathcal{T}_q$ and $\mathcal{T}_s$. We then compute directional keyword overlap:
$$\text{Overlap}(s, q) = \frac{|\mathcal{T}_q \cap \mathcal{T}_s|}{|\mathcal{T}_q|}$$
Sentences satisfying $\text{Overlap}(s, q) \le \tau$ are retained and concatenated to form the filtered context $C_{\text{filtered}}$. If all sentences exceed $\tau$, the context is retained unchanged to prevent catastrophic empty inputs. $C_{\text{filtered}}$ is then ingested by the reader model to extract the start and end span logits.

---

## 7. Experimental Setup
* **Model Checkpoint:** `deepset/roberta-base-squad2` (124,056,578 parameters, PyTorch).
* **Dataset:** $N=50$ curated answerable validation instances from `rajpurkar/squad_v2`.
* **Hardware & Runtime:** Apple Silicon MPS / Google Colab T4 GPU, batch size = 1.
* **Evaluation Protocol:** Standard SQuAD Exact Match (EM) and Macro-averaged Token F1 computed via the official Hugging Face `evaluate` library.
* **Controlled Conditions:** Condition A (Clean Gold Context), Condition B (Noisy Unfiltered), Condition C (Noisy Filtered, $\tau=0.3$).
* **Ablation Range:** $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$.

---

## 8. Results
The controlled experimental results are summarized in Table 1:

**Table 1: Controlled Experimental Results across Grounding Conditions**
| Experimental Condition | Exact Match (EM) | Token F1 Score | Net Delta vs. Condition B |
|---|:---:|:---:|:---:|
| **Condition A: Clean Baseline** | **68.00%** | **73.17%** | — |
| **Condition B: Noisy Unfiltered** | **54.00%** | **60.98%** | Base Noisy Level |
| **Condition C: Noisy Filtered ($\tau=0.3$)** | **16.00%** | **16.14%** | **-38.00% EM** (Collateral Collapse) |

### Predicted vs. Actual Outcome
We predicted a positive recovery of $+8.00\%$ to $+12.00\%$ EM (reaching $62.00\%\text{--}66.00\%$). In reality, the filter caused EM to plummet to **16.00%**, generating a **$-38.00\%$ negative recovery**. The empirical data definitively refutes the hypothesis.

---

## 9. Ablation Study
To evaluate whether threshold miscalibration caused the failure, we ablated $\tau$ from $0.1$ to $0.5$:

**Table 2: Threshold Sensitivity Ablation Study**
| Threshold ($\tau$) | Exact Match (EM) | Token F1 Score | Qualitative Failure Mode |
|:---:|:---:|:---:|---|
| **0.1** | **14.00%** | **14.14%** | Hyper-aggressive: excises almost all sentences |
| **0.2** | **12.00%** | **12.14%** | Severe context destruction |
| **0.3** | **16.00%** | **16.14%** | Primary experimental setting |
| **0.4** | **20.00%** | **21.48%** | Partial recovery as fewer sentences are deleted |
| **0.5** | **30.00%** | **32.78%** | Best threshold, but still 24.00% below unfiltered |

The ablation curve reveals a monotonic upward trajectory as $\tau$ relaxes. However, even at $\tau=0.5$, performance ($30.00\%$ EM) remains far below the unfiltered noisy condition ($54.00\%$), proving that threshold adjustment cannot fix the fundamental flaw of lexical filtering.

---

## 10. Failure Analysis
An instance-level outcome audit across all $50$ test instances categorizes transitions from Condition B to Condition C:
* **Filter Helped ($B_{\text{wrong}} \rightarrow C_{\text{correct}}$):** **3 cases (6.0%)**
* **Filter Made No Difference ($B \rightarrow C$ invariant):** **25 cases (50.0%)** (20 both wrong, 5 both correct)
* **Filter Hurt ($B_{\text{correct}} \rightarrow C_{\text{wrong}}$):** **22 cases (44.0%)**

In $44.0\%$ of cases, the filter inflicted catastrophic collateral damage. For instance, in the query *"When were the Normans in Normandy?"*, the model correctly extracted *"10th and 11th centuries"* under the noisy unfiltered condition. The filter flagged the authentic gold sentence because it contained both `normans` and `normandy`, deleted the evidence, and forced the model to output an empty prediction.

---

## 11. Discussion
The empirical results provide a critical negative finding for RAG system design. In knowledge-intensive question answering, causal evidence sentences inherently mirror query vocabulary. Consequently, **lexical overlap is an intrinsically confounded metric**: it cannot separate malicious surface distractors from genuine evidentiary support. An unguided token-overlap filter behaves like an over-aggressive immune response, destroying the host tissue (gold context) while attempting to eliminate the pathogen (spurious distractor). To protect RAG readers at inference time without retraining, future architectures must employ Natural Language Inference (NLI) entailment scoring or cross-encoder rerankers that evaluate semantic support rather than token intersection.

---

## 12. Limitations
This study has five primary limitations: (1) Scoped sample size ($N=50$) provides indicative rather than statistically significant results; (2) Evaluation focused on a single synthetic high-overlap template rather than the full five-dimension SURE taxonomy; (3) The rule-based filter lacks semantic embeddings or directional entailment modeling; (4) Experiments utilized an extractive encoder (`roberta-base`) rather than a generative autoregressive LLM; and (5) Ablation thresholds were swept on the evaluation set without an independent held-out split.

---

## 13. Conclusion
We investigated the vulnerability of Retrieval-Augmented Language Models to spurious grounding features and evaluated a zero-shot, sentence-level token-overlap filter. While we verified the paper's core finding that query-correlated noise degrades reader performance by $14.00\%$ EM, our pre-registered filtering hypothesis was conclusively disproven: the filter reduced performance to $16.00\%$ EM due to a $44.0\%$ collateral deletion rate of authentic gold evidence. This negative result establishes that surface lexical overlap is an unsuitable criterion for RAG noise defense, demonstrating that robust grounding verification requires deep semantic reasoning.

---

## 14. References
1. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.
2. Y. Lyu, Z. Tan, H. Wu, et al., "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models," in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2024.
3. S. Perçin, X. Su, Q. S. Syed, P. Howard, A. Kuvshinov, L. Schwinn, and K.-U. Scholl, "Investigating the Robustness of Retrieval-Augmented Generation at the Query Level," in *Proceedings of the 4th Workshop on Generation, Evaluation and Metrics (GEM²)*, ACL, 2025, arXiv:2507.06956.
4. N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the Middle: How Language Models Use Long Contexts," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 12, pp. 157–173, 2024.
5. S. Yang, J. Wu, W. Ding, N. Wu, S. Liang, M. Gong, H. Li, H. Zhang, A. X. Chang, and D. Zhang, "Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data," in *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2026, arXiv:2503.05587.
