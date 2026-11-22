# RAG Robustness & Reliability
## NLP Research Project — Research Track
**B.Tech Computer Science & AI | Rishihood University (Newton School of Technology)**

> Investigating the vulnerability of Retrieval-Augmented Language Models to spurious features in grounding data, and evaluating the effectiveness of inference-time lexical filtering as a mitigation strategy.

---

## 📄 Anchor Paper
**Title:** Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data  
**Venue:** ACL 2026 | **arXiv:** [2503.05587](https://arxiv.org/abs/2503.05587)  
**Authors:** Shiping Yang et al. (SFU, Microsoft, Atlassian)

---

## 👥 Team
| Name | Enrollment | Role / Focus Area |
|------|------------|-------------------|
| Siddhartha Shukla | 230104 | Lead Researcher, Experimental Pipeline, IEEE Paper |
| Piyush Kaushal | 230112 | Baseline Reproduction, SQuAD Data Evaluation |
| Sugat Athawale | 230109 | Hypothesis Formulation, Ablation Sweep Design |
| Ayush Dev | 230061 | Failure Analysis, 2x2 Outcome Matrix, Error Taxonomies |

---

## 🔬 Key Results

| Condition | Exact Match | Token F1 |
|-----------|-------------|----------|
| Clean Baseline | 68.00% | 73.17% |
| Spurious Noisy | 54.00% | 60.98% |
| Token-Overlap Filtered | 16.00% | 16.14% |

**Finding:** A 14% robustness drop under spurious grounding is confirmed. A zero-shot Jaccard token-overlap filter was hypothesised to recover this drop but instead caused catastrophic degradation (−38% EM) due to 44% collateral deletion of gold evidence sentences — a significant negative result demonstrating that lexical overlap is insufficient as a spurious detection signal.
