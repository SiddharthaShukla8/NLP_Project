# NLP Project M1 + M2 — Complete Antigravity Agent Prompt
## RAG Robustness & Reliability | Research Track
---

> **HOW TO USE THIS:** Copy the prompt under each phase heading and paste it directly into Antigravity. Run them in order. Do not skip phases. Each prompt builds on the output of the previous one.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 1 — Research Area Framing & Paper Search Strategy
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am a B.Tech Computer Science & AI student doing an NLP Research Project under the Research Track. My chosen research area is "RAG Robustness & Reliability." My anchor paper is:

Title: "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data"
Venue: ACL 2026

I need you to do the following:

1. Explain in depth what "RAG (Retrieval-Augmented Generation)" means — how it works, what the retriever does, what the language model does, how the grounding data is used, and why robustness matters in this pipeline.

2. Explain what "spurious features in grounding data" means in the context of NLP and RAG. Give concrete examples of what a spurious feature looks like in a retrieved passage.

3. Give me a search strategy to find 4 more related papers in this area. Provide:
   - Exact search queries I should type into Google Scholar, Semantic Scholar, and ACL Anthology
   - 4 specific papers you recommend I look for (title + authors + year + venue + why they are relevant to RAG robustness)
   - What I should look for in each paper to fill my comparison table

4. Tell me the key concepts I must understand before I can read my anchor paper deeply: RAG architecture, retrieval noise types, faithfulness vs. factuality in LLMs, and robustness evaluation in NLP.

Be detailed. I will use this output to guide my entire literature survey.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 2 — Anchor Paper Deep Read Template
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am reading my anchor paper for my NLP Research Project (Research Track):

Title: "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026)

Help me extract every piece of information I need from this paper by giving me a detailed reading guide. For each section of the paper (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Limitations), tell me:

1. What specific information I must extract and write down
2. What questions I should be able to answer after reading that section
3. What terms or concepts I need to look up if I don't understand them

Then give me a structured extraction template I should fill while reading. It must cover:
- What is the exact research problem this paper addresses?
- What gap in prior work does it identify?
- What is their proposed method or approach?
- What dataset(s) do they use? (name, size, domain, source)
- What model(s) do they use? (name, size, whether it's pretrained/fine-tuned)
- What are the evaluation metrics? (EM, F1, custom robustness score, etc.)
- What are the main quantitative results they report?
- What ablations or analysis experiments do they run?
- What do they claim as their main contribution?
- What limitations do they acknowledge?
- What future directions do they suggest?
- Is code publicly available? (GitHub, HuggingFace, etc.)
- Is the dataset publicly available?
- What hardware and compute do they use?

After giving me this template, explain what I should look for specifically to assess whether this paper is REPRODUCIBLE by me on a free Colab/Kaggle GPU.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 3 — 5-Paper Comparison Table Builder
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am doing an NLP Research Project (Research Track) in the area of RAG Robustness & Reliability. I have read 5 papers and need to produce a structured comparison table as my M1 deliverable.

My 5 papers are:
- Paper 1 (Anchor): "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" — ACL 2026
- Paper 2: "Investigating the Robustness of RAG at the Query Level" — 2025
- Paper 3: [I will fill this — a foundational RAG paper like REALM or RAG by Lewis et al.]
- Paper 4: [I will fill this — a paper on retrieval noise or irrelevant passage effects]
- Paper 5: [I will fill this — a paper on reranking, context ordering, or faithfulness in RAG]

Do the following:

1. Give me the exact Markdown table structure I must fill, with these exact columns:
   Research Problem | Method/Approach | Dataset | Model | Evaluation Metric | Code Available | Data Available | Main Finding | Key Limitation | Possible Extension

2. For my anchor paper (Paper 1), fill every column with your best knowledge. Mark anything uncertain as [VERIFY IN PAPER].

3. For Paper 2 ("Investigating the Robustness of RAG at the Query Level"), fill every column with your best knowledge. Mark anything uncertain as [VERIFY IN PAPER].

4. For Papers 3, 4, 5 — give me placeholder rows with guiding questions in each cell so I know what to write when I read those papers.

5. After the table, write a 3-paragraph synthesis that: (a) identifies the common research thread across all 5 papers, (b) points out how methods and datasets differ, and (c) states what gap none of them fully address — this gap will become my research problem in M3.

Format everything in clean Markdown. I will copy this directly into my project report.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 4 — Anchor Paper Justification + Research Problem Statement
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am writing M1 of my NLP Research Project (Research Track). I need to write two things:

(A) ANCHOR PAPER JUSTIFICATION (~200 words)
Write a formal justification paragraph for why I selected "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026) as my anchor paper. The justification MUST cover all four required criteria:
1. Scientifically meaningful — why this problem matters in the current NLP landscape
2. (Partially) reproducible — what specific part of the paper I can reproduce on free compute (Colab/Kaggle)
3. Computationally feasible — why the hardware requirements are within my reach as a student
4. Extensible — what research gap I see in this paper that I can investigate independently in M3 and M4

(B) INITIAL RESEARCH PROBLEM STATEMENT (2–3 sentences)
Write a clear, specific research problem statement that:
- Names the phenomenon being studied (RAG robustness to spurious grounding features)
- States what is currently unknown or understudied
- Hints at the direction of my independent investigation

Rules for both:
- Use academic tone but no jargon without explanation
- Do not make claims I can't verify yet
- Be specific — no vague statements like "this is an important area"
- The justification and problem statement must be consistent with each other

Give me final, submission-ready text. Also give me a checklist to verify I haven't missed anything required by the M1 guidelines.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 5 — Reproducibility Checklist (M1 Final Item)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am completing M1 of my NLP Research Project. My final item is a Reproducibility Checklist for my anchor paper:

Title: "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026)

The checklist has exactly these 10 items from my project guidelines:
1. Public dataset available?
2. Dataset size documented?
3. Public code available?
4. Public pretrained model available?
5. Hardware requirement known?
6. Training time known?
7. Dependencies listed?
8. Evaluation metric clearly defined?
9. Feasible baseline runnable on Colab/Kaggle/free GPU?
10. Are reported results sufficiently reproducible?

Do the following:

1. For each checklist item, explain what exactly I need to look for in the paper to answer it — what section, what text to search for, what to check on GitHub/HuggingFace/Papers With Code.

2. Give me a formatted checklist table with: Item | Answer (Yes/No/Partial/Unknown) | Evidence/Notes — where I should fill the Answer and Evidence columns after checking the paper.

3. Tell me what to do for EACH possible scenario:
   - If code is NOT publicly available → what is my alternative plan for M2?
   - If the dataset requires licensing → what is my alternative public dataset?
   - If the model is too large for Colab → what is my smaller substitute model?
   - If training time is unknown → how do I estimate it myself?

4. Based on what you know about this paper and ACL 2026 papers generally, give me your best guess for each checklist item and flag which ones I should verify most carefully.

5. Give me the URLs I should check:
   - ACL Anthology page for this paper
   - GitHub search query to find the paper's code
   - Hugging Face search query
   - Papers With Code URL

Format the final checklist in clean Markdown, submission-ready.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 6 — M1 Complete Assembly & Final Review
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I have completed all components of M1 for my NLP Research Project (Research Track). Before I submit, I need you to help me assemble and review the entire M1 package.

My M1 must contain exactly:
1. Research area name and description
2. Five paper references (title, authors, venue, year)
3. Structured comparison table (10 columns, 5 papers)
4. Initial research problem statement
5. Anchor paper selection with written justification
6. Filled reproducibility checklist

Do the following:

1. Give me a complete M1 document structure in Markdown with proper headings, so I can paste all my content into the right sections. Include a cover line at the top: "M1 — Literature Survey | NLP Research Project | Research Track | RAG Robustness & Reliability"

2. Write a final M1 cover paragraph (~100 words) that introduces what this submission contains and what research direction it establishes. This goes at the very top after the title.

3. Give me a self-review checklist — 15 specific questions I should answer YES to before submitting. Example: "Does the justification mention all 4 required criteria (scientific meaningfulness, reproducibility, feasibility, extensibility)?" Make all 15 questions this specific.

4. Tell me the 3 most common mistakes students make in M1 of a research-track NLP project and how to avoid each one.

5. Tell me what I should already be thinking about for M2 based on what I wrote in M1, so I can start preparing while M1 is being reviewed.
```

---
---
## ═══════════════════════════════════════════════════
## M2 PHASE — BASELINE REPRODUCTION
## ═══════════════════════════════════════════════════
---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 7 — Baseline Scoping: What Exactly to Reproduce
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am starting M2 of my NLP Research Project (Research Track). My anchor paper is:

"Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026)

My research area is RAG Robustness & Reliability. In M2, I need to "reproduce a feasible baseline (not necessarily the entire paper)."

Do the following:

1. Explain what "reproducing a feasible baseline" means in the context of this paper. What is the minimum reproduction that is academically honest and sufficient for M2?

2. Identify the specific experiment from this paper that is MOST reproducible for a student with:
   - Free Colab T4 GPU (16GB VRAM) or Kaggle P100
   - Access to HuggingFace models and datasets
   - No budget for OpenAI API calls (must use open-source models)
   - Time budget of 3–5 days for M2

3. Define my reproduction target clearly:
   - Exact dataset to use (name, HuggingFace ID if available, number of examples I should test on)
   - Exact model to use (name, HuggingFace ID, size in parameters)
   - Exact metric to compute (EM, F1, robustness score — whichever is in the paper)
   - What "clean baseline" means vs. "noisy/spurious condition" in this paper

4. Explain the concept of "spurious features" that this paper injects — what does the injected noise look like concretely? Give me 3 examples of what a question + clean context + spurious context might look like.

5. Tell me what the paper reports as their baseline result so I know what number I am trying to reproduce.

6. Give me a "scope decision" table:
   | Component | Full Paper Version | My Feasible Version | Why I Scoped It |

This decision table will go into my M2 documentation to justify what I chose to reproduce and why.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 8 — Dataset Documentation
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am working on M2 of my NLP Research Project. I need to fully document the dataset I am using for my baseline reproduction of a RAG robustness paper.

My dataset is likely one of the standard open-domain QA datasets used in RAG research (Natural Questions, TriviaQA, or similar — based on what my anchor paper uses).

Do the following:

1. Give me a complete Dataset Documentation template with these exact fields:
   - Dataset Name
   - Source / Citation
   - HuggingFace Dataset ID (if available)
   - Task Type (e.g., open-domain QA, extractive QA)
   - Domain (e.g., Wikipedia-based, web-based)
   - Language
   - Total Size (number of examples)
   - Train Split Size
   - Validation Split Size
   - Test Split Size
   - Input Format (what the model receives)
   - Output Format (what the model should produce)
   - Evaluation Metric Used
   - Known Data Quality Issues
   - License / Usage Restrictions
   - How the paper uses this dataset
   - How I am using this dataset (subset? full?)
   - Why I chose this dataset / or why the paper chose it

2. For Natural Questions (NQ) AND TriviaQA, fill the above template completely so I have reference answers for the two most likely datasets.

3. Explain how the dataset is modified in a RAG robustness experiment:
   - What is the "clean" version of the dataset?
   - How are spurious passages added/injected?
   - What does a single data example look like before and after spurious injection?

4. Give me the exact Python code to load whichever dataset is used, inspect the first 3 examples, and print the keys/fields so I understand the data structure:

```python
from datasets import load_dataset
# Fill this in for NQ and TriviaQA
```

5. Tell me what to document about my data loading process so my M2 submission is complete.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 9 — Preprocessing Pipeline Documentation
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am working on M2 of my NLP Research Project (Research Track, RAG Robustness). I need to document my preprocessing pipeline completely.

My setup:
- Dataset: [Natural Questions or TriviaQA — whichever my anchor paper uses]
- Model: A BERT-based or smaller open-source QA model from HuggingFace
- Task: Reproducing a clean baseline (no spurious features yet) and then a noisy condition (with spurious passage injected)
- Environment: Google Colab with T4 GPU

Do the following:

1. Write a complete numbered preprocessing pipeline for a RAG QA robustness experiment. Cover every step from raw data → model input:
   Step 1: Load dataset
   Step 2: Filter/subset (if needed)
   Step 3: [continue through all steps...]
   Final Step: Batch and run inference

2. For each step, explain:
   - What the input looks like
   - What the output looks like
   - What decisions I made and why (e.g., max_length=512, truncation='longest_first')
   - What could go wrong here

3. Explain what a "spurious passage injection" step looks like in preprocessing. How do I construct the noisy input? How is it different from the clean input? Write the preprocessing step for this specifically.

4. Give me the complete tokenization code for a QA model on this dataset:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")

def preprocess_example(example, spurious_passage=None):
    # Your complete tokenization logic here
    pass
```

5. Tell me exactly what I need to document about my preprocessing in the M2 submission. Give me a checklist of preprocessing documentation items the marker will expect to see.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 10 — Baseline Implementation (Full Colab Notebook)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am implementing the baseline for M2 of my NLP Research Project (Research Track).

My goal: Reproduce a clean RAG QA baseline — a model answering questions with relevant retrieved passages, NO spurious noise injected yet — and compute its Exact Match (EM) and F1 score on a held-out validation set.

My setup:
- Dataset: Natural Questions (or TriviaQA) from HuggingFace
- Model: deepset/roberta-base-squad2 (or similar small open-source QA model)
- Environment: Google Colab T4 GPU
- Framework: HuggingFace Transformers + Datasets + Evaluate

Write me a COMPLETE, RUNNABLE Google Colab notebook in Python. The notebook must have these exact sections as markdown headers:

## Section 1: Environment Setup
- Install all dependencies with exact version pins
- Mount Google Drive if needed

## Section 2: Dataset Loading & Inspection
- Load dataset
- Print shape, columns, first 3 examples
- Document what each field contains

## Section 3: Model & Tokenizer Loading
- Load model and tokenizer
- Print model size (number of parameters)
- Move to GPU

## Section 4: Preprocessing
- Tokenize examples
- Handle long contexts (truncation, stride)
- Create DataLoader

## Section 5: Baseline Inference (Clean Condition)
- Run inference on validation set (use first 500 examples to stay within Colab time limits)
- Store predictions

## Section 6: Metric Computation
- Compute Exact Match and F1
- Print final scores clearly

## Section 7: Results Table
- Print a formatted table: Metric | Original Paper Result | My Reproduced Result | Gap

## Section 8: Observations
- Print sample predictions (5 examples): question | context | gold answer | predicted answer | correct?

Write the full code for every section. Use comments in the code explaining every non-obvious decision. The notebook must run top-to-bottom without errors on Colab free tier.

Also: after the notebook, give me a "Hyperparameters Table" I need to fill and submit:
| Hyperparameter | Value Used | Paper's Value (if stated) |
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 11 — Spurious Noise Injection (The Noisy Condition)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am extending my M2 baseline for my NLP Research Project. After running the clean baseline, I now need to also run the "noisy condition" — where spurious passages are injected into the grounding context — to reproduce the robustness result from my anchor paper.

My anchor paper: "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026)

Do the following:

1. Explain the concept of "spurious features in grounding data" precisely. What exactly is being injected, and what makes it "spurious" vs. just "irrelevant"? Give 3 concrete examples showing:
   - The original question
   - The clean relevant passage
   - The injected spurious passage
   - Why the spurious passage is problematic for the model

2. Write a Python function that takes a clean QA example and injects a spurious passage in the way the paper describes:

```python
def inject_spurious_passage(example, spurious_type="lexical_overlap"):
    """
    Takes a clean QA example and injects a spurious passage.
    
    Args:
        example: dict with 'question', 'context', 'answers'
        spurious_type: type of spurious feature to inject
    
    Returns:
        Modified example with spurious passage injected
    """
    pass
```

3. Write the complete modified inference loop that:
   - Takes the clean dataset
   - Injects spurious passages for each example
   - Runs the model on the noisy input
   - Computes EM and F1 on the noisy condition

4. Write the final comparison table I need to produce:
| Condition | EM | F1 | Notes |
|-----------|----|----|-------|
| Original Paper — Clean | [paper's number] | [paper's number] | |
| Original Paper — Noisy | [paper's number] | [paper's number] | |
| My Reproduction — Clean | [my number] | [my number] | |
| My Reproduction — Noisy | [my number] | [my number] | |
| Reproduction Gap — Clean | | | |
| Reproduction Gap — Noisy | | | |

5. Explain what the "robustness drop" metric means — the difference between clean and noisy performance — and why it is the central finding of the paper.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 12 — Reproduction Gap Analysis (The Most Important Part of M2)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I have completed my baseline reproduction for M2 of my NLP Research Project. I now need to analyze and document the reproduction gap — the difference between the paper's reported results and my reproduced results.

My results:
- Paper's clean baseline EM: [I will fill this]
- My clean baseline EM: [I will fill this]
- Paper's noisy condition EM: [I will fill this]
- My noisy condition EM: [I will fill this]

Do the following:

1. Give me a complete framework for analyzing a reproduction gap in NLP. List every possible source of gap with a clear explanation:
   - Random seed differences
   - Dataset version/split differences
   - Preprocessing differences
   - Model version/checkpoint differences
   - Compute precision differences (fp16 vs fp32, GPU vs CPU)
   - Partial reproduction (I tested on fewer examples)
   - Implementation differences (no public code available)
   - Evaluation metric implementation differences

2. For each gap source, tell me:
   - How I can TEST whether this is causing the gap
   - What evidence I need to document
   - How to report it honestly in my submission

3. Write the reproduction gap analysis section I need to submit. It must be in this format:
   
   **3a. Observed Gap:**
   [What the numbers show]
   
   **3b. Likely Sources of Gap:**
   [Ranked list of most likely causes with evidence]
   
   **3c. What I Was Able to Control:**
   [What I kept the same as the paper]
   
   **3d. What I Could Not Control:**
   [What differed due to compute/access constraints]
   
   **3e. Interpretation:**
   [Does this gap invalidate my reproduction? Why or why not?]

4. Write a formal "Reproduction Observations" section (~300 words) that is submission-ready. It must:
   - Acknowledge the gap honestly
   - Explain it scientifically
   - Show that I understand WHY the gap exists
   - Not treat the gap as a failure
   - Connect observations to what I will investigate in M3

5. Tell me the 3 most important sentences in any reproduction gap analysis — the ones that show the marker I understand research methodology.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROMPT 13 — M2 Complete Assembly & Final Review
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I am assembling my final M2 submission for my NLP Research Project (Research Track). My M2 must contain exactly these deliverables according to my project guidelines:

1. Dataset documentation
2. Preprocessing pipeline
3. Baseline implementation (notebook/code)
4. Original vs. reproduced results (comparison table)
5. Reproduction observations (gap analysis)

Additionally from the guideline: "document dataset, preprocessing, model, hyperparameters, training setup, metrics, original result, reproduced result, and the reproduction gap."

Do the following:

1. Give me the complete M2 document structure in Markdown with all required sections and subsections as headings. Title: "M2 — Baseline Reproduction | NLP Research Project | Research Track | RAG Robustness & Reliability"

2. For each section, tell me exactly how long it should be and what it must contain. Example: "Dataset Documentation — 1 table + 1 paragraph. Must include: name, source, size, format, splits, metric, license."

3. Write a submission-ready Hyperparameters Table:
| Hyperparameter | Value I Used | Paper's Value | Source |
|---|---|---|---|
| Model name | | | |
| Model size | | | |
| Max sequence length | | | |
| Batch size | | | |
| Random seed | | | |
| Evaluation subset size | | | |
| GPU used | | | |
| Framework version | | | |

4. Give me a M2 self-review checklist — 12 specific questions I must answer YES to before submitting.

5. Write the bridge paragraph that connects M2 to M3 — what my reproduction experience revealed that motivates my research gap and hypothesis. This is the most important intellectual move in the whole project: show that my M3 gap came from what I learned in M2, not from random selection.

6. Tell me the single most important thing my M2 submission needs to demonstrate to get full marks, according to the marking philosophy: "Marks reward research thinking, experimental discipline, scientific reasoning, critical analysis, honest reproducibility reporting."
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## BONUS PROMPT — Teaser for M3 (Start Thinking Now)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
I have completed M1 (literature survey) and M2 (baseline reproduction) of my NLP Research Project on RAG Robustness & Reliability. My anchor paper is "Quantifying and Improving the Robustness of Retrieval-Augmented LMs Against Spurious Features in Grounding Data" (ACL 2026).

I want to start thinking about M3 (Research Gap + Original Hypothesis) without jumping ahead. Help me think through the gap identification process.

Do the following:

1. List 5 potential research gaps in RAG robustness that a student could realistically investigate with free compute:
   - For each gap, state: what is unknown, why it matters, what experiment could test it, and what result I would predict

2. For each gap, evaluate it on:
   - Scientific novelty (has anyone studied this exactly?)
   - Computational feasibility (Colab T4 GPU, 3–5 days)
   - Testability (can I write a clear hypothesis in the M3 format?)
   - Connection to my anchor paper (does it extend what the paper found?)

3. Give me the M3 hypothesis format template so I know what I'm building toward:
   "Changing X will improve/decrease Y because Z. We predict approximately ___ improvement/change."

4. Tell me: while I am running M2 experiments, what observations should I be writing down that will directly help me write M3? What patterns in model behavior, failure cases, or surprising results should I note?

5. Recommend the ONE research gap I should pursue in M3 based on what is most feasible, most scientifically interesting, and most directly connected to what the anchor paper investigated.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## QUICK REFERENCE: Run Order & What Each Prompt Produces
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Prompt | When to Run | What It Produces | M1 or M2 Item |
|--------|-------------|------------------|---------------|
| 1 | Day 1 — Before reading any paper | Background understanding + search strategy | M1 Setup |
| 2 | Day 1–2 — While reading anchor paper | Paper extraction template (fill while reading) | M1 Setup |
| 3 | Day 2–3 — After reading all 5 papers | 5-paper comparison table + synthesis | M1 Deliverable |
| 4 | Day 3 — After comparison table | Anchor paper justification + research problem | M1 Deliverable |
| 5 | Day 3 — Final M1 item | Reproducibility checklist + alternative plans | M1 Deliverable |
| 6 | Day 3 — Before M1 submission | Full M1 document assembly + self-review | M1 Final |
| 7 | Day 4 — Starting M2 | Reproduction scope decision table | M2 Setup |
| 8 | Day 4 — Environment setup | Dataset documentation template (filled) | M2 Deliverable |
| 9 | Day 4–5 — Building pipeline | Preprocessing pipeline (numbered + code) | M2 Deliverable |
| 10 | Day 5–6 — Running experiments | Full Colab notebook (clean baseline) | M2 Deliverable |
| 11 | Day 6 — After clean baseline | Noisy condition code + comparison table | M2 Deliverable |
| 12 | Day 7 — After all experiments | Reproduction gap analysis (written) | M2 Deliverable |
| 13 | Day 7 — Before M2 submission | Full M2 assembly + self-review | M2 Final |
| Bonus | Day 7 — While wrapping M2 | M3 gap ideas + hypothesis direction | M3 Prep |

