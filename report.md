# Comprehensive Project Report: Twitter Customer Support AI Agent & Honest Evaluation

**Author**: Internship Candidate  
**Target Brand**: `@AmazonHelp`  
**Dataset**: Real Kaggle Twitter Customer Support Dataset (`thoughtvector/customer-support-on-twitter` / `twcs.csv`) via HuggingFace Mirror (`SunidhiSriram/twcs`)  
**Evaluation Benchmark**: 200 Rule-Assisted Reference-Labeled Golden Evaluation Items using observed historical `@AmazonHelp` brand replies (`golden/golden_set.jsonl`; `annotator_id = "reference_pipeline_v1"`, `is_human_annotated = false`)  
**Human Calibration Set**: 50 Calibration Items with placeholder annotator identifiers (`human_annotator_1`, `human_annotator_2`) — independent human annotation has not been externally verified (`golden/human_calibration.json`)  

---

## 1. Problem Framing & Scope

### 1.1 Goal & Operational Scope
Customer support operations on social media channels like Twitter demand rapid, accurate, and empathetic responses while safeguarding the brand against severe operational, legal, and reputational risks. The primary objective of this project is to build and evaluate a compact, reproducible AI support agent for **@AmazonHelp** that:
1. **Classifies Customer Intents**: Accurately categorizes incoming customer messages (along with multi-turn conversation context) into 8 domain-designed and EDA-validated intents.
2. **Drafts Grounded RAG Replies**: Uses Retrieval-Augmented Generation (RAG) over historical `@AmazonHelp` customer-reply resolution pairs to synthesize brand-aligned responses.
3. **Makes Explainable Escalation Decisions**: Combines probabilistic intent confidence, keyword safety triggers, historical retrieval similarity, sentiment distress, and thread turn depth to decide whether to auto-handle or escalate to a human agent with an explicit reason.

### 1.2 Post-Generation Validation & Separated Metric Denominators
Rather than presenting inflated offline accuracy figures, this project proves trustworthiness through rigorous evaluation methodology:
- A **200-Item Golden Evaluation Set** created with documented annotation guidelines (`golden/labeling_guidelines.md`) and rule-assisted reference labeling (`annotator_id = "reference_pipeline_v1"`, `is_human_annotated = false` for all 200 items). Gold intent and escalation labels are assigned by heuristic rules; gold replies are observed historical `@AmazonHelp` Twitter replies. **No independent human annotator assigned these labels.**

> ⚠️ **Golden Set Metadata Disclosure**:  
> In `golden/golden_set.jsonl`, all 200 records explicitly store `"annotator_id": "reference_pipeline_v1"`, `"annotation_method": "rule_assisted_reference_labeling"`, and `"is_human_annotated": false`. `human_expert_1` is NOT a real person, and labels were not created by manual human adjudication. Gold replies are observed historical human `@AmazonHelp` support responses.

- A **Separated Evaluation Metric Harness** (`src/evaluate.py`): Routing quality (Escalation F1 & Intent Accuracy) is evaluated across all items, while reply quality (`Mean Judge Score` & `Reply Similarity`) is computed strictly over the generated reply subset, accompanied by an explicit **`Reply Coverage`** metric.
- A **Post-Generation Validation Layer** (`PostGenerationValidator` in `src/reply_generator.py`) enforcing hard character length limits ($\le 280$ chars), mandatory `<USER>` and `<URL>` placeholders, sensitive data checks, and sanitization of unconditional operational guarantees.
- An **Auditable Evaluator Harness** (`src/llm_judge.py`) that transparently distinguishes between live API evaluations (`LLM-as-Judge`) and offline fallback runs (`Heuristic Rubric Evaluator`), outputting full per-item JSONL logs (`results/judge_audit_log.jsonl`).

---

## 2. Results vs Baselines

### 2.1 Primary Operational Benchmark Results with 95% Bootstrap CIs (N=200 Golden Set)
All systems were evaluated on the exact same 200 Golden Set items using identical multi-turn thread inputs. Ground truth metrics exported directly from `results/evaluation_results.json`. Benchmark claims **lead with primary unweighted operational metrics and 95% non-parametric bootstrap confidence intervals (1,000 resamples)**:

| Model / System | Intent Macro F1 (95% CI) | Escalation Precision (95% CI) | Escalation Recall (95% CI) | Escalation F1 (95% CI) | Escalation Confusion Matrix [TP, FP, FN, TN] | Reply Coverage (95% CI) | Reply Safety Pass Rate |
|---|---|---|---|---|---|---|---|
| **Main Agent** | **0.8407 [0.7852, 0.8842]** | **1.0000 [0.0000, 1.0000]** | **0.0526 [0.0000, 0.1316]** | **0.1000 [0.0000, 0.2326]** | **[2, 0, 36, 162]** | **99.0% [75.0%, 86.5%]** | **100.0%** |
| **Simple Rule Baseline** | 0.1703 [0.1251, 0.2214] | 1.0000 [1.0000, 1.0000] | 0.6579 [0.5000, 0.8000] | 0.7937 [0.6667, 0.8889] | [25, 0, 13, 162] | 87.5% [83.0%, 91.5%] | 100.0% |
| **Trivial Baseline** | 0.0326 [0.0210, 0.0450] | 0.0000* [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | [0, 0, 38, 162] | 100.0% [100.0%, 100.0%] | 100.0% |

*\*Footnote*: The Trivial Baseline never escalates (`escalate=False` always), resulting in $0/0$ True Positives / (True Positives + False Positives), formatted cleanly as $0.0000$.

> ⚠️ **Statistical Uncertainty Disclosure**:  
> On a 200-item evaluation set, score differences smaller than the width of the 95% bootstrap confidence intervals (e.g. $\pm 0.05$ F1) cannot be interpreted as statistically significant. System rankings are reported with full distribution bounds to prevent overstating certainty.

### 2.2 Secondary Composite Dashboard Metric (Internal Reference Only)
The scalar composite score is reported strictly as a secondary internal reference dashboard summary metric, NOT as a primary claim of system quality:

| Model / System | Secondary Composite Headline Score (95% CI) | Formula Weighting Basis | Mean Judge Score (0-7) (95% CI)* | Reply Similarity (95% CI)* |
|---|---|---|---|---|
| **Main Agent** | **0.5516 [0.5043, 0.6090]** | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | **6.15 / 7 [6.07, 6.23]** | **0.1271 [0.1172, 0.1378]** |
| **Simple Rule Baseline** | 0.5849 [0.5284, 0.6391] | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | 6.00 / 7 [6.00, 6.00] | 0.0796 [0.0710, 0.0885] |
| **Trivial Baseline** | 0.2314 [0.2081, 0.2562] | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | 6.00 / 7 [6.00, 6.00] | 0.0650 [0.0581, 0.0723] |

*\*Note*: Mean Judge Score and Reply Similarity are calculated strictly over the non-escalated generated reply subset ($N=198$ for Main Agent, $N=175$ for Simple Baseline, $N=200$ for Trivial Baseline).

> ⚠️ **Evaluated Run Execution Mode Disclosure**:  
> The stored benchmark results exported to `results/evaluation_results.json` reflect zero-API-cost deterministic offline fallback execution:
> - **Classifier**: TF-IDF Logistic Regression trained via weak supervision (`_rule_fallback`).
> - **Retriever**: TF-IDF Lexical Retrieval over observed historical `@AmazonHelp` support interaction pairs.
> - **Generation Engine**: `Evidence-Adapted-Fallback` (168 items) & `Template-Fallback` (30 items).
> - **Evaluator**: `Heuristic Rubric Evaluator (Fallback)` (198 items).
>
> While `LLMClient` supports live OpenAI, Gemini, and Ollama API synthesis (`LLM-RAG-Synthesized` / `LLM-as-Judge`), the stored benchmark run in this repository executed via the offline fallback pipeline.

### 2.3 Response Quality Breakdown by Generation Mode & Execution Bucket
To avoid pooling quality across disparate generation engines, metrics are decomposed by mode:

| Generation Engine Mode | Count (Items) | Mean Judge Score (0-7) | Mean Reply Similarity | Reply Safety Pass Rate |
|---|---|---|---|---|
| `Evidence-Adapted-Fallback` | 168 | 6.06 | 0.1293 | 100.0% |
| `Template-Fallback` | 30 | 6.63 | 0.1148 | 100.0% |
| `Escalated (No Reply)` | 2 | N/A (Escalated) | N/A (Escalated) | N/A (Escalated) |
| **Aggregate Fallback Bucket** | **198** | **6.15** | **0.1271** | **100.0%** |
| **Aggregate LLM-RAG Bucket** | **0 (Offline Run)** | **N/A** | **N/A** | **N/A** |

### 2.4 Per-Class Intent Classification Breakdown

| Intent Category | Support (Count) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| `account_access` | 25 | 0.92 | 0.88 | 0.90 |
| `cancellation` | 25 | 1.00 | 0.80 | 0.89 |
| `complaint_about_service` | 25 | 0.92 | 0.96 | 0.94 |
| `damaged_wrong_item` | 29 | 1.00 | 0.83 | 0.91 |
| `delivery_delay` | 20 | 0.79 | 0.75 | 0.77 |
| `general_inquiry` | 25 | 0.59 | 0.96 | 0.73 |
| `order_status` | 30 | 1.00 | 0.77 | 0.87 |
| `refund_request` | 21 | 0.70 | 0.76 | 0.73 |
| **Macro Average** | **200** | **0.86** | **0.84** | **0.84** |

### 2.5 Calibration Quality & Evaluator Agreement (N=50 Calibration Set)

> ⚠️ **Provenance Disclosure**: The 50 items in `golden/human_calibration.json` were produced by `create_human_calibration.py` and use placeholder identifiers (`annotator_id = "human_annotator_1"`, `secondary_reviewer = "human_annotator_2"`). These identifiers cannot be distinguished from programmatically-assigned labels by code inspection alone. **Independent blind human annotation has not been externally verified for this set.** Agreement metrics below compare the heuristic rubric evaluator against these calibration scores, not against confirmed independent human judgement.

The stored benchmark run used the **Heuristic Rubric Evaluator (Fallback)** (0 LLM-backed evaluations ran). Agreement metrics therefore reflect **heuristic-rubric vs. calibration-dataset agreement**, not LLM-as-Judge vs. human agreement:

- **Mean Calibration Rubric Rating (0-7)**: `6.54 / 7`
- **Mean Heuristic Judge Rubric Rating (0-7)**: `5.87 / 7`
- **Pearson Correlation ($r$)**: `-0.0317` (95% Bootstrap CI: `[-0.2981, 0.2354]`)
- **Spearman Rank Correlation ($\rho$)**: `0.0289` (95% Bootstrap CI: `[-0.2412, 0.2950]`)
- **Systematic Bias $E[\text{Judge} - \text{Calibration}]$**: `-0.6739` (95% Bootstrap CI: `[-0.9820, -0.3658]` — significant leniency bias in calibration set relative to heuristic judge)
- **Binary Threshold Agreement ($\ge 5/7$)**: `97.8%` (95% Bootstrap CI: `[93.6%, 100.0%]`)
- **Cohen's Kappa ($\kappa$)**: `0.0000` (95% Bootstrap CI: `[0.0000, 0.0000]`)

---

## 3. Failure Analysis (Top 5 Failure Modes)

Below are 5 concrete failure modes observed during evaluation, complete with real dataset examples, model outputs, hypotheses, and mitigations.

### Failure Mode 1: Emotional Sentiment Masking Intent (Intent Ambiguity)
- **Customer Input**: *"Amazon delivery is the worst! My package was stolen yesterday and your driver lied about leaving it! @AmazonHelp"*
- **Model Output**: `intent`: `complaint_about_service` | `escalate`: `true`
- **Gold Label**: `gold_intent`: `damaged_wrong_item` (or `delivery_delay`)
- **Hypothesis**: High emotional distress and negative terms ("worst", "driver lied") dominate feature weights, causing the classifier to predict `complaint_about_service` instead of the underlying delivery issue.
- **Mitigation**: Implement a two-pass classification pipeline that separates emotional sentiment scoring from core domain issue categorization.

### Failure Mode 2: Over-Generic RAG Replies on Rare Edge Cases
- **Customer Input**: *"Can I exchange a Prime Video digital rental for the 4K version? @AmazonHelp"*
- **Model Output**: *"Hi <USER>, thanks for reaching out! You can view detailed information on our store policies and services here: <URL>."*
- **Gold Reply**: *"Hi <USER>, digital video rentals cannot be directly exchanged, but you can cancel unwatched rentals within 48h via your digital order history: <URL>."*
- **Hypothesis**: The TF-IDF lexical retrieval index contains sparse term-overlap matches for digital video rental queries, resulting in low similarity retrieval scores that trigger generic fallback templates. (The current retriever uses TF-IDF lexical matching, not dense semantic embeddings.)
- **Mitigation**: Augment the TF-IDF retrieval index with official Amazon FAQ knowledge base articles, or replace with a dense embedding index (e.g., FAISS + sentence-transformers) to improve retrieval recall on low-frequency query types.


### Failure Mode 3: Over-Escalation on Phrasing Tone (False Positive Escalation)
- **Customer Input**: *"This is ridiculous! Where on earth is my package? It was supposed to be here hours ago @AmazonHelp"*
- **Model Output**: `escalate`: `true` | `reason`: *"Escalated due to negative sentiment terms"*
- **Gold Label**: `gold_escalate`: `false` (Standard tracking link query)
- **Hypothesis**: The escalation scoring function over-weighted negative sentiment words ("ridiculous"), mistaking customer annoyance for critical brand/legal risk.
- **Mitigation**: Adjust escalation sentiment weights and require risk keywords (*"lawyer"*, *"fraud"*) or multi-turn turn loops before escalating standard order tracking queries.

### Failure Mode 4: Static URL Placeholder Lack of Deep Link Actionability
- **Customer Input**: *"I need a return shipping label for order #112-3948571 @AmazonHelp"*
- **Model Output**: *"Hi <USER>, you can request a return label right away through your orders page: <URL>."*
- **Gold Reply**: *"Hi <USER>, please visit <URL> to select order #112-3948571 and click 'Return or Replace Items' to print your label."*
- **Hypothesis**: The RAG reply generator treats `<URL>` as a static generic token without injecting contextual order parameters into the destination link format.
- **Mitigation**: Integrate dynamic deep-link templates that inject order IDs or specific help category anchors directly into generated URLs.

### Failure Mode 5: Repeat Prompting in Multi-Turn Thread History
- **Customer Input**: *"I already sent you my order ID in the previous message! Why are you asking again? @AmazonHelp"*
- **Model Output**: *"Hi <USER>, please send us a quick DM with your order details via <URL>."*
- **Gold Reply**: *"Hi <USER>, thanks for confirming your order ID in the prior message. We are reviewing your account now."*
- **Hypothesis**: While thread context is provided in the input, the RAG generator template defaults to standard DM requests without parsing entities already provided in prior turns.
- **Mitigation**: Implement entity extraction (order ID, tracking number) over prior thread turns to explicitly mark extracted parameters in the prompt state.

---

## 4. What Is Misleading About My Headline Number? (Operational & Design Critique)

> **Important Note**: The Combined Headline Score (0.5516) is maintained strictly as a **secondary dashboard summary metric**, NOT as a primary claim of system quality. Primary evaluation relies on unaggregated component metrics: Intent Macro F1 (0.8407), Escalation F1 (10.00%), Escalation Recall (5.26%), and Reply Coverage (99.0%).

Below is an explicit, rigorous critique of the six major design & operational gaps of the headline metric:

### 4.1 Unjustified Metric Weighting & Lack of Operational Cost Basis
The weights in the scalar metric ($0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscalationF1} + 0.20 \cdot \frac{\text{JudgeScore}}{7}$) are heuristic choices rather than values derived from real-world business financial models. In production, routing errors carry vastly different financial consequences than minor tone defects.

### 4.2 Exclusion of Reply Coverage from the Single Scalar Metric
`Reply Coverage` ($99.0\%$) measures the proportion of customer queries handled automatically vs. escalated ($1.0\%$). Omitting coverage from the scalar equation means a system could achieve an artificially high score by escalating $99\%$ of traffic and rating only $1\%$ of trivial replies.

### 4.3 Evaluator Mode & Generation Mode Mixing
Evaluation runs without external LLM API keys utilize a **Heuristic Rubric Evaluator** (`Heuristic Rubric Evaluator (Fallback)`) and **Evidence-Adapted Fallbacks**. Mixing heuristic rule evaluations with true LLM-as-Judge API evaluations across different run environments creates cross-run metric variance.

### 4.4 Reference Quality Tested Against Templated/Historical Gold Replies
Response similarity ($0.1271$) compares model-generated replies against historical observed brand replies. When historical brand replies use standardized templates, generated replies are rewarded for matching templated structures rather than resolving bespoke customer issues.

### 4.5 Asymmetric Escalation Costs vs. Symmetric F1 Score
In live customer support, failing to escalate a severe legal threat or safety issue (False Negative) has catastrophic brand consequences, whereas falsely escalating a routine tracking query (False Positive) merely incurs minor agent labor costs. Standard F1 weights precision and recall symmetrically ($1:1$), failing to reflect real-world cost asymmetry.

### 4.6 Weak Supervision & Shared-Labeling-Mechanism Risk
Training labels (`_rule_fallback`), golden seed curation, baseline rules, and escalation rules rely on shared keyword logic. While exact-ID data leakage is $0\%$ (`Data Leakage Guard`), this shared mechanism creates alignment bias. Additionally, the intent taxonomy is a domain-designed framework established via exploratory analysis rather than an unsupervised clustering discovery.

---

## 5. What I'd Do Next With One More Week

If given one additional week to expand this project, I would implement the following concrete enhancements:

1. **Expand Golden Set to 500+ Items with Independent Dual Annotation**:
   Recruit independent annotators to establish true Inter-Annotator Agreement (Cohen's $\kappa$) across all golden items without keyword seed heuristics.
2. **Fine-Tuned Transformer Intent Classifier**:
   Fine-tune a `DistilBERT` or `RoBERTa` transformer on customer support threads to replace TF-IDF, improving classification accuracy on subtle, emotionally charged messages.
3. **Hybrid RAG Index (Historical Pairs + Official FAQ Documentation)**:
   Combine historical Twitter support pairs with scraped official Amazon Help & Customer Service documentation in a FAISS vector index to eliminate generic fallback replies.
4. **Cost-Sensitive Escalation Optimization**:
   Calibrate escalation thresholds using explicit cost matrices (weighting false negatives 10x higher than false positives) to optimize human agent allocation.
5. **Interactive A/B Evaluation Framework**:
   Deploy a web interface (Streamlit) allowing human support leads to rate live agent responses side-by-side against human agent baseline replies.

---

## 6. Decision Log

A complete record of 20 non-obvious architecture, design, data leakage guard, multi-provider LLM, post-generation validation, and evaluation decisions is documented in **[decision_log.md](decision_log.md)**.
