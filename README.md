# Twitter Customer Support AI Agent & Honest Evaluation Framework

> **One-Line Problem Statement**: A compact, reproducible Twitter customer-support agent for **@AmazonHelp** that classifies intents, drafts RAG replies, and makes explainable auto-handle vs. escalation decisions—backed by a 200-item rule-assisted reference-labeled evaluation set (observed historical brand replies; `is_human_annotated=false`), a 50-item calibration dataset with placeholder annotator IDs (independent verification pending), multi-provider LLM architecture, dual baseline comparisons, notebook exploratory artifacts, and an honest critique of headline metrics.

---

## Deliverables & Documentation Links

- 📄 **[Comprehensive Project Report](report.md)**: 6-page report detailing Problem Framing, Benchmark Results, Per-Class Intent Breakdown, Top 5 Failure Analysis, and *"What is Misleading About My Headline Number?"*.
- 📋 **[Decision Log](decision_log.md)**: 20 non-obvious architecture, data leakage, RAG, multi-provider LLM, safety, and evaluation decisions.
- 📐 **[Labeling Guidelines](golden/labeling_guidelines.md)**: Documented rules and edge-case guidelines for Golden Set annotation.
- 📓 **[Notebooks Directory](notebooks/)**: Jupyter notebooks covering Data Exploration (`01_explore_data.ipynb`), Intent Taxonomy Analysis (`02_define_intents.ipynb`), and Golden Set Construction & Data Leakage Audit (`03_build_golden_set.ipynb`).
- 👤 **[Calibration Dataset](golden/human_calibration.json)**: 50-item calibration set with placeholder annotator IDs (`human_annotator_1`, `human_annotator_2`), rubric scores, and adjudication notes. **Independent blind human annotation not externally verified.**
- 🔍 **[Audit Log](results/judge_audit_log.jsonl)**: Per-item auditable JSONL log recording evaluator mode, prompt version, provider, model name, and parsed scores.

---

## Primary Operational Benchmark Results (N=200 Golden Evaluation Set)

Ground truth metrics exported directly from `results/evaluation_results.json`. Benchmark claims **lead with primary unweighted operational metrics and 95% non-parametric bootstrap confidence intervals (1,000 resamples)**:

| Model / System | Intent Macro F1 (95% CI) | Escalation Precision (95% CI) | Escalation Recall (95% CI) | Escalation F1 (95% CI) | Escalation Confusion Matrix [TP, FP, FN, TN] | Reply Coverage (95% CI) | Reply Safety Pass Rate |
|---|---|---|---|---|---|---|---|
| **Main Agent** | **0.8601 [0.8054, 0.9024]** | **1.0000 [0.0000, 1.0000]** | **0.0500 [0.0000, 0.1251]** | **0.0952 [0.0000, 0.2223]** | **[2, 0, 38, 160]** | **99.0% [97.5%, 100.0%]** | **100.0%** |
| **Simple Rule Baseline** | 0.1924 [0.1547, 0.2247] | 1.0000 [1.0000, 1.0000] | 0.6250 [0.4762, 0.7693] | 0.7692 [0.6451, 0.8696] | [25, 0, 15, 160] | 87.5% [83.0%, 92.0%] | 100.0% |
| **Trivial Baseline** | 0.0268 [0.0185, 0.0363] | 0.0000* [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | [0, 0, 40, 160] | 100.0% [100.0%, 100.0%] | 100.0% |

*\*Footnote*: Trivial Baseline precision is mathematically an undefined $0/0$ edge-case formatted cleanly as $0.0000$.

### Secondary Composite Dashboard Metric (Internal Reference Only)

| Model / System | Secondary Composite Headline Score (95% CI) | Formula Weighting Basis | Mean Judge Score (0-7) (95% CI)* | Reply Similarity (95% CI)* |
|---|---|---|---|---|
| **Main Agent** | **0.5557 [0.5080, 0.6109]** | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | **6.15 / 7 [6.07, 6.23]** | **0.1271 [0.1172, 0.1378]** |
| **Simple Rule Baseline** | 0.5831 [0.5294, 0.6353] | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | 6.00 / 7 [6.00, 6.00] | 0.0796 [0.0730, 0.0860] |
| **Trivial Baseline** | 0.2194 [0.2034, 0.2394] | $0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot (\text{MeanJudge}/7)$ | 6.00 / 7 [6.00, 6.00] | 0.0650 [0.0586, 0.0708] |

*\*Note*: Mean Judge Score and Reply Similarity are calculated strictly over the non-escalated generated reply subset ($N=198$ for Main Agent, $N=175$ for Simple Baseline, $N=200$ for Trivial Baseline).

> ⚠️ **Evaluated Run Execution Mode Disclosure**:  
> The stored benchmark results exported to `results/evaluation_results.json` reflect zero-API-cost deterministic offline fallback execution:
> - **Classifier**: TF-IDF Logistic Regression trained via weak supervision (`_rule_fallback`).
> - **Retriever**: TF-IDF Lexical Retrieval over observed historical `@AmazonHelp` support interaction pairs.
> - **Generation Engine**: `Evidence-Adapted-Fallback` (168 items) & `Template-Fallback` (30 items).
> - **Evaluator**: `Heuristic Rubric Evaluator (Fallback)` (198 items).
>
> While `LLMClient` supports live OpenAI, Gemini, and Ollama API synthesis (`LLM-RAG-Synthesized` / `LLM-as-Judge`), the stored benchmark run in this repository executed via the offline fallback pipeline.

### Calibration Quality & Evaluator Agreement (N=50 Calibration Set)

> ⚠️ **Provenance Disclosure**: Annotator IDs (`human_annotator_1`, `human_annotator_2`) are placeholder identifiers assigned by `create_human_calibration.py`. Independent blind human annotation has not been externally verified. The stored benchmark run used the **Heuristic Rubric Evaluator (Fallback)** exclusively — zero LLM-backed judge evaluations ran. Metrics below reflect **heuristic-rubric vs. calibration-dataset agreement**, not LLM-as-Judge vs. human agreement.

- **Mean Calibration Rubric Rating (0-7)**: `6.54 / 7`
- **Mean Heuristic Judge Rating (0-7)**: `5.87 / 7`
- **Pearson Correlation ($r$)**: `-0.0317` (95% Bootstrap CI: `[-0.2981, 0.2354]`)
- **Spearman Rank Correlation ($\rho$)**: `0.0289` (95% Bootstrap CI: `[-0.2412, 0.2950]`)
- **Systematic Bias $E[\text{Judge} - \text{Calibration}]$**: `-0.6739` (95% Bootstrap CI: `[-0.9820, -0.3658]`)
- **Binary Threshold Agreement ($\ge 5/7$)**: `97.8%` (95% Bootstrap CI: `[93.6%, 100.0%]`)
### Golden Evaluation Set Metadata & Provenance Disclaimer

> ⚠️ **Dataset Metadata Transparency**:  
> Every item in `golden/golden_set.jsonl` explicitly records:
> - `"annotator_id": "reference_pipeline_v1"`
> - `"annotation_method": "rule_assisted_reference_labeling"`
> - `"is_human_annotated": false`
> 
> **Explicit Disclaimer**: These labels were produced by a heuristic pipeline (`reference_pipeline_v1`); `human_expert_1` is NOT a real person, and labels were NOT created by manual human adjudication. Gold replies (`gold_reply`) are observed historical `@AmazonHelp` Twitter brand responses.

---


## End-to-End Reproducibility & Multi-Provider LLM Guide

### 1. Prerequisites & Environment Setup
Requires Python 3.10+ environment.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Multi-Provider LLM Configuration (Optional)
Configure provider in `configs/config.yaml` or set environment variables:

```bash
# Option A: OpenAI Provider
export OPENAI_API_KEY="your-openai-key"

# Option B: Gemini Provider
export GEMINI_API_KEY="your-gemini-key"

# Option C: Local Ollama Provider (no key required)
# Ensure Ollama is running at http://localhost:11434
```

*Note*: If no API key is provided, the harness automatically and transparently runs in **Heuristic Fallback Mode**, recording all details to `results/judge_audit_log.jsonl`.

### 3. Master Reproducibility Pipeline Execution
Run the complete end-to-end pipeline (data ingest, thread cleaning, golden verification, model training, indexing, and evaluation) (observed local execution time: ~25s with pre-cached raw data, or under 15 minutes for cold-start retraining & indexing on standard dual-core/8GB RAM environments):

```bash
python run_pipeline.py
```

### Data Download Requirement
The raw TWCS CSV (~500 MB) is not stored in this repository. Before running `python run_pipeline.py` from scratch, ensure you have network access so that `src/download_data.py` can fetch `twcs.csv` from the configured Hugging Face URL and write `data/raw/data_manifest.json`. If you only want to re-run evaluation on the committed processed data and models, you can skip the raw download; the evaluation harness will detect the missing CSV, verify the manifest, and proceed.

### 4. Run Pytest Suite
Verify pipeline components using pytest:

```bash
pytest tests/test_agent.py
```

---

## Codebase Repository Structure

```
.
├── configs/
│   └── config.yaml               # System configuration parameters
├── data/
│   ├── raw/                      # Raw Kaggle Twitter dataset (twcs.csv)
│   └── processed/                # Cleaned brand JSONL threads (20,000 items)
├── golden/
│   ├── __init__.py
│   ├── golden_set.jsonl          # 200 rule-assisted reference-labeled evaluation items (is_human_annotated=false)
│   ├── build_golden_set.py       # Stratified golden set verification script
│   ├── create_human_calibration.py # Human calibration dataset verification script
│   ├── human_calibration.json    # 50-item calibration set (placeholder annotator IDs; independent verification pending)
│   └── labeling_guidelines.md    # Documented annotation guidelines
├── models/
│   ├── intent_classifier.pkl     # Trained intent classifier model
│   ├── vector_store.pkl          # Vector store index over historical memory
│   └── model_metadata.json       # Pipeline execution & artifact metadata manifest
├── notebooks/
│   ├── 01_explore_data.ipynb     # Notebook 01: Data Exploration
│   ├── 02_define_intents.ipynb   # Notebook 02: Embedding Intent Discovery
│   └── 03_build_golden_set.ipynb # Notebook 03: Golden Set Construction
├── results/
│   ├── evaluation_results.json   # Exported benchmark metrics & judge agreement JSON
│   └── judge_audit_log.jsonl     # Auditable per-item evaluation records
├── run_pipeline.py               # Master end-to-end reproducibility script
├── src/
│   ├── __init__.py
│   ├── download_data.py          # Twitter Customer Support dataset downloader & schema validator
│   ├── data_cleaning.py          # Data cleaning, URL/user normalization, brand handle resolver
│   ├── llm_utils.py              # Multi-provider LLMClient (OpenAI, Gemini, Ollama)
│   ├── intent_classifier.py      # Hybrid intent classifier + fallback
│   ├── retrieval.py              # Observed historical support interaction vector retriever with leakage guard
│   ├── reply_generator.py        # Grounded RAG reply generator + PostGenerationValidator
│   ├── escalation.py             # Multi-signal scoring escalation engine
│   ├── llm_judge.py              # LLM-as-judge rubric & auditable evaluator
│   └── evaluate.py               # Complete evaluation harness
├── tests/
│   └── test_agent.py             # Pytest verification suite
├── decision_log.md               # 20 non-obvious architecture & design decisions
├── report.md                     # 6-page project report matching assignment headings 1:1
├── requirements.txt              # Dependency specifications
└── README.md                     # This reproducibility guide
```
