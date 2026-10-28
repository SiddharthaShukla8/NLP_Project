# M1 — Literature Survey & Anchor Paper Selection
## NLP Research Project | Research Track | B.Tech Computer Science & Artificial Intelligence
**Research Area:** RAG Robustness & Reliability  
**Anchor Paper:** *Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data* (arXiv:2503.05587, ACL 2026)  
**Authors of Anchor Paper:** Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang  
**Author:** Siddhartha Shukla  
**Repository:** [https://github.com/SiddharthaShukla8/NLP_Project](https://github.com/SiddharthaShukla8/NLP_Project)

---

## Executive Summary & Cover Statement
This document constitutes the formal Milestone 1 (M1) deliverable for the Research Track of the 4th-Year Undergraduate Natural Language Processing (NLP) course. It establishes the theoretical and empirical foundation for investigating the robustness of Retrieval-Augmented Generation (RAG) language models against spurious features and retrieval noise. In this milestone, we conduct an extensive literature survey across 5 pivotal papers in the RAG reliability space, synthesize key methodological differences and research frontiers in a 10-column structured comparison matrix, provide a rigorous 4-pillar academic justification for selecting our anchor paper (*Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data*, arXiv:2503.05587, ACL 2026), formulate our initial research problem statement, and establish a fully verified 10-point Reproducibility Checklist to ensure computational feasibility on accessible academic compute (Google Colab / Kaggle T4 GPUs).

---

## 1. Research Area Framing & Conceptual Foundations

### 1.1 What is Retrieval-Augmented Generation (RAG)?
Retrieval-Augmented Generation (RAG) is a hybrid architectural paradigm designed to overcome the fundamental limitations of parametric Large Language Models (LLMs)—namely knowledge cutoff dates, domain-specific hallucinations, untraceable reasoning, and excessive parameter scaling costs. 

A standard RAG pipeline operates through three distinct yet tightly coupled stages:
1. **The Retriever ($\mathcal{R}$):** Given an input query $q$, the retriever searches an external non-parametric corpus $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ using either dense dual-encoders (e.g., DPR, Contriever, BGE) or sparse lexical indexers (e.g., BM25). It returns a ranked subset of top-$k$ grounding documents $\mathcal{D}_k = \{d_{(1)}, \dots, d_{(k)}\}$.
2. **The Grounding / Context Aggregator:** The retrieved passages are concatenated with the original query $q$ into a structured grounding prompt:
   $$\mathbf{x} = [\text{Instruction}; d_{(1)}; \dots; d_{(k)}; q]$$
3. **The Generator / Reader ($\mathcal{G}$):** An autoregressive or sequence-to-sequence language model generates the output token sequence conditioning on both its internal parametric weights and the explicit non-parametric context.

### 1.2 What are "Spurious Features in Grounding Data"?
In natural language processing and statistical learning, a **spurious feature** is an input attribute or artifact that correlates with answers or alters model confidence, but has no genuine causal or semantic relationship to the underlying truth.

In the anchor paper (*Yang et al., 2025*), spurious features are categorized into a formal taxonomy across grounding passages:
1. **Style Perturbations:** Linguistic register, tone, or surface phrasing changes that do not alter the semantic claim.
2. **Source Artifacts:** Surface markers implying spurious authority (e.g., *"Official archives confirm that..."*).
3. **Logic Shifts:** Inverted premises or superficially plausible deductions that mislead attention.
4. **Format Perturbations:** Markdown headers, HTML snippets, or JSON structures in grounding documents.
5. **Metadata Perturbations:** False temporal stamps or publication headers that trick models into prioritizing distractor passages.

---

## 2. Comprehensive Literature Survey (5 Related Papers)

1. **Paper 1 (Anchor Paper):**
   * *Title:* Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data
   * *Authors:* Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hongzhi Li, Hengyuan Zhang, Angel X. Chang, Dongmei Zhang
   * *Venue & Identifier:* arXiv:2503.05587 / ACL 2026
   * *Core Theme:* Proposes the SURE evaluation framework and SIG benchmark to quantify how RALMs fail under implicit spurious features (style, source, logic, format, metadata) and introduces synthetic mitigation strategies.
2. **Paper 2:**
   * *Title:* Investigating the Robustness of Retrieval-Augmented Generation at the Query Level
   * *Authors:* Sezen Perçin, Xin Su, Qutub Sha Syed, Phillip Howard, Aleksei Kuvshinov, Leo Schwinn, Kay-Ulrich Scholl
   * *Venue & Identifier:* Proceedings of the Fourth Workshop on Generation, Evaluation and Metrics (GEM²), ACL 2025 (ACL Anthology: `2025.gem-1.38`, arXiv:2507.06956)
   * *Core Theme:* Systematically investigates the vulnerability of RAG retrievers (BM25, BGE-base, Contriever) and generators (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B) to query-level noise across 1,092 controlled experiments.
3. **Paper 3 (Foundational RAG Benchmark):**
   * *Title:* Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
   * *Authors:* Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
   * *Venue & Year:* NeurIPS 2020
   * *Core Theme:* Foundational architecture combining parametric seq2seq models (BART) with non-parametric dense retrieval (DPR).
4. **Paper 4 (Noise & Distraction Analysis):**
   * *Title:* Benchmarking Large Language Models in Retrieval-Augmented Generation (CRUD-RAG)
   * *Authors:* Y. Lyu, Z. Tan, H. Wu, et al.
   * *Venue & Year:* ACL 2024
   * *Core Theme:* Characterization of retrieval noise into irrelevant, contradictory, and counterfactual grounding passages.
5. **Paper 5 (Context Ordering & Position Bias):**
   * *Title:* Lost in the Middle: How Language Models Use Long Contexts
   * *Authors:* Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
   * *Venue & Year:* Transactions of the Association for Computational Linguistics (TACL) 2024
   * *Core Theme:* Empirical evaluation showing performance degrades dramatically when relevant information is situated in the middle of retrieved contexts rather than at the boundaries.

---

## 3. Structured 10-Column Comparison Matrix

| Research Problem | Method / Approach | Dataset | Model | Evaluation Metric | Code Available | Data Available | Main Finding | Key Limitation | Possible Extension |
|---|---|---|---|---|---|---|---|---|---|
| **Paper 1 (Anchor):** Robustness Against Spurious Features in Grounding Data (ACL 2026 / arXiv:2503.05587) | SURE framework: systematic injection of style, source, logic, format, and metadata perturbations into grounding data; data synthesis training. | SIG Benchmark (SIG_Wiki, SIG_Trivial) | Mistral-7B, Llama-3.1-8B, Llama-3-70B, Qwen-2.5-7B, GPT-4o | Robustness Ratio (RR), Lose Rate (LR), Win Rate (WR) | Yes (`anonymous.4open.science/r/RAG-SpuriousFeatures-62B3` & GitHub) | Yes (SIG benchmark splits) | RALMs exhibit pronounced vulnerability to non-semantic spurious features (RR drops significantly across models); synthetic training boosts resilience. | Evaluated primarily on large LLMs (>7B params); requires heavy compute to execute end-to-end. | Reproducing core distraction effects on lightweight open-source QA models and developing inference-time filters. |
| **Paper 2:** Investigating Robustness of RAG at Query Level (ACL GEM² 2025 / arXiv:2507.06956) | Evaluates end-to-end RAG vulnerability when user queries undergo typographical errors, ambiguity, redundancy insertion, and formal tone modifications across 1,092 experiments. | Natural Questions (NQ), HotpotQA, BioASQ | Retrievers: BM25, BGE-base, Contriever; Generators: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B | Hit@k, MRR@k, Exact Match (EM), F1, Robustness Degradation ($\Delta$) | Yes (Open-source research artifacts / GitHub) | Yes (Standard public benchmark splits) | Text retrievers undergo severe recall collapse under subtle query perturbations; dense retrievers resist redundancy better but are highly fragile to typos compared to BM25. | Evaluates query-side perturbations exclusively; assumes retrieved candidate documents contain no adversarial spurious features. | Combining query perturbation resilience with grounding passage spurious feature filtering into a unified robust pipeline. |
| **Paper 3 (Foundational):** Retrieval-Augmented Generation for Knowledge-Intensive Tasks (NeurIPS 2020) | Dual-encoder DPR retriever coupled with pre-trained BART generator, trained end-to-end with marginalization over latent retrieved docs. | Natural Questions, TriviaQA, CuratedTREC, MS-COCO | DPR (Dual BERT) + BART-large (400M params) | Exact Match (EM), BLEU-1, Rouge-L | Yes (HuggingFace Transformers `RagToken` / `RagSequence`) | Yes (Standard NLP QA corpora) | Combining non-parametric retrieval memory with parametric generation sets SOTA on open-domain QA without hallucinating facts. | Fails completely if DPR returns noisy or adversarial passages; vulnerable to retrieval errors. | Adding retrieval verification and passage confidence filtering mechanisms. |
| **Paper 4:** Benchmarking Large Language Models in Retrieval-Augmented Generation (ACL 2024) | Framework categorizing retrieval noise into: (1) Irrelevant, (2) Inaccurate, (3) Counterfactual, and (4) Multiple-conflict passages. | CRUD-RAG benchmark, NQ, PopQA | GPT-4, LLaMA-2-70B, ChatGLM, Baichuan | ROUGE-L, F1, Hallucination Rate, Rejection Rate | Yes (GitHub project repository) | Yes (HuggingFace Hub) | Even state-of-the-art LLMs struggle to ignore counterfactual noise when it overlaps lexically with the prompt, dropping accuracy by >30%. | Evaluates closed commercial API models without releasing open-source fine-tuning checkpoints. | Extending noise classification to quantify how position and length of noise modulate error severity. |
| **Paper 5:** Lost in the Middle: How Language Models Use Long Contexts (TACL 2024) | Multi-document QA benchmark varying the position of the gold document among $K \in \{10, 20, 30\}$ distractor documents. | Natural Questions Multi-document QA, TriviaQA | MPT-30B-Instruct, Llama-2-7B-32k, Claude-1.3, GPT-3.5-Turbo | Exact Match (EM), Accuracy vs. Golden Document Index | Yes (GitHub repository `nelson-liu/lost-in-the-middle`) | Yes (Pre-processed JSONL splits) | Performance is U-shaped: models retrieve facts accurately at the start or end of context, but drop up to 40% when gold context is in the middle. | Focuses on context position rather than active spurious correlation or lexical distractor content. | Investigating interaction effects between "lost-in-the-middle" position and lexical spurious noise. |

---

## 4. Multi-Paper Synthesis & Research Landscape Analysis

Across all surveyed papers, a fundamental limitation of modern LLM reasoning is revealed: **Language models conditioned on retrieved context are easily distracted by superficial non-causal surface patterns.** 

While foundational papers like Lewis et al. (Paper 3) assumed retrieval would monotonically improve factual generation, subsequent studies demonstrated that models are brittle to query perturbations (Paper 2), context positioning (Paper 5), general irrelevant documents (Paper 4), and structured spurious features in grounding text (Anchor Paper 1). 

The anchor paper establishes that even 70B parameter models fail to ignore irrelevant stylistic and metadata cues. This reveals an open research opportunity: **How can resource-constrained open-source models be protected against spurious grounding features using lightweight, inference-time verification?**

---

## 5. Anchor Paper Justification & Initial Problem Statement

### 5.1 Anchor Paper Justification (~200 Words)
We select *"Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data"* (arXiv:2503.05587, ACL 2026) as our primary anchor paper based on four rigorous academic criteria:

1. **Scientifically Meaningful:** The paper moves beyond simplistic random noise to address realistic, implicit spurious features (style, source, logic, format, metadata) that occur in enterprise and web RAG pipelines.
2. **(Partially) Reproducible:** The core empirical claim—that models degrade when non-causal context matches surface query features—is reproducible using open-source models and standard datasets without requiring proprietary APIs.
3. **Computationally Feasible:** While the authors evaluate up to 70B parameter models, the underlying attention-distraction phenomenon can be benchmarked and investigated on compact, open-source reader architectures (`deepset/roberta-base-squad2`) on accessible academic compute (Google Colab T4 GPU).
4. **Extensible:** The paper highlights vulnerabilities across diverse feature categories, opening clear pathways for investigating inference-time filtering and confidence calibration mechanisms in M3–M5.

### 5.2 Initial Research Problem Statement
*Current Retrieval-Augmented Generation (RAG) systems suffer substantial accuracy degradation when grounding contexts contain non-causal spurious features that exhibit surface correlations with user queries.* 
*While existing work introduces the SURE framework to benchmark this on multi-billion parameter LLMs, it remains unexplored whether lightweight inference-time filtering can mitigate this vulnerability in compute-constrained reader architectures.* 
*Our investigation aims to empirically reproduce this degradation using accessible open-source models and evaluate compute-efficient filtering mechanisms to restore answer robustness.*

---

## 6. Reproducibility Checklist (10 Verification Items)

| # | Checklist Item | Evaluation | Evidence & Practical Notes |
|---|---|---|---|
| 1 | **Public dataset available?** | **YES** | SIG benchmark is open-sourced; standard open-domain benchmarks (SQuAD v2.0, NQ) are hosted on Hugging Face Datasets Hub. |
| 2 | **Dataset size documented?** | **YES** | SQuAD v2 validation split has 11,873 instances; feasible reproduction scopes to $N=50$ held-out answerable instances. |
| 3 | **Public code available?** | **YES** | Anonymous repository available at `anonymous.4open.science/r/RAG-SpuriousFeatures-62B3` and GitHub `maybenotime/RAG-SpuriousFeatures`. |
| 4 | **Public pretrained model available?** | **YES** | Evaluated LLMs (Mistral-7B, Llama-3.1-8B) and our scoped reader (`deepset/roberta-base-squad2`) are public on Hugging Face. |
| 5 | **Hardware requirement known?** | **YES** | Paper uses multi-A100 clusters for 70B models. Our scoped baseline executes on a single free Google Colab T4 GPU (<4 GB VRAM). |
| 6 | **Training time known?** | **YES / NA** | Inference on $N=50$ validation examples completes in under 30 seconds on GPU. |
| 7 | **Dependencies listed?** | **YES** | Python 3.10+, `torch`, `transformers`, `datasets`, `evaluate`, `accelerate`. |
| 8 | **Evaluation metric clearly defined?** | **YES** | Paper defines Robustness Ratio (RR), Lose Rate (LR), and Win Rate (WR). Our scoped baseline measures Exact Match (EM) and Token F1 via `evaluate.load('squad')`. |
| 9 | **Feasible baseline runnable on Colab/Kaggle free GPU?** | **YES** | Confirmed: Runnable notebook tested and verified. |
| 10 | **Are reported results sufficiently reproducible?** | **YES** | The core phenomenon (performance degradation under spurious grounding features) is empirically verified. |
