# Decision Log: 15 Non-Obvious Architecture & Design Decisions

This document details 15 non-obvious engineering decisions made during the design, implementation, and evaluation of the Twitter Customer Support AI Agent.

---

### 1. Locked Brand Selection to `@AmazonHelp` with Dynamic Metadata Resolution
- **Decision**: Locked the primary brand handle to `@AmazonHelp` and saved `resolved_brand_handle` to `data/processed/brand_metadata.json`.
- **Rationale**: Prevents cross-brand contamination by ensuring downstream RAG retrieval, prompt formatting, and evaluation components load the exact brand handle resolved at data cleaning time.

### 2. Multi-Turn Thread Reconstruction Over Single Tweet Parsing
- **Decision**: Built a conversation graph parser to group tweets by `conversation_id` and `in_response_to_tweet_id` into multi-turn threads (`customer_message` + up to 3 prior turns).
- **Rationale**: Customer intent and escalation risks frequently depend on prior conversation context (e.g. repeated unhelpful turns). Single-tweet evaluation creates severe context blindness.

### 3. Explicit `conversation_id` Data Leakage Guard
- **Decision**: Enforced an explicit `conversation_id` exclusion filter across both the historical RAG vector memory index construction and intent classifier model training.
- **Rationale**: Any thread present in the 200 Golden Evaluation Set is strictly scrubbed from historical retriever memory and training samples. (Note: Near-duplicate phrasings in Twitter support data represent an unmonitored evaluation boundary beyond exact ID matching).

### 4. Domain-Designed Intent Taxonomy & Exploratory Validation
- **Decision**: Established an 8-intent taxonomy (`order_status`, `delivery_delay`, `refund_request`, `damaged_wrong_item`, `account_access`, `cancellation`, `complaint_about_service`, `general_inquiry`) based on domain analysis of Twitter customer support operations for `@AmazonHelp` and exploratory dataset inspection.
- **Rationale**: Reflects real-world customer problem distributions and business resolution pathways. (Note: The taxonomy is domain-designed and validated through EDA rather than automatically discovered via unsupervised clustering).

### 5. Multi-Provider LLM Client with Explicit Endpoints & Fallback Logging
- **Decision**: Implemented an explicit `LLMClient` (`src/llm_utils.py`) supporting `openai`, `gemini`, and `ollama` endpoints with detailed metadata logging (`provider`, `model_name`, `used_fallback`, `error`).
- **Rationale**: Allows seamless execution across cloud APIs and local Ollama deployments while logging fallback usage transparently when API keys are unconfigured.

### 6. Chronological $N$-Turn Context Representation
- **Decision**: Formatted thread context as the last $N$ prior turns (up to 3 prior turns) concatenated chronologically with explicit role tags (`Customer: ...` / `Brand: ...`).
- **Rationale**: Providing structured role boundaries prevents the classifier and LLM from confusing historic brand replies with current customer complaints.

### 7. Multi-Signal Escalation Scoring Function
- **Decision**: Used a 6-signal weighted risk formula ($0.30 \cdot \text{IntentRisk} + 0.25 \cdot \text{Uncertainty} + 0.20 \cdot \text{RiskKeywords} + 0.10 \cdot \text{LowRetrievalSim} + 0.08 \cdot \text{ThreadDepth} + 0.07 \cdot \text{Sentiment}$) with hard keyword triggers (*"lawyer"*, *"sue"*, *"fraud"*).
- **Rationale**: Machine learning classifiers alone miss rare high-impact safety triggers. Wiring historical retrieval similarity score (`low_retrieval_sim`) directly into escalation catches unfamiliar edge cases.

### 8. Restricting RAG Retrieval Memory to Historical Customer-Reply Pairs
- **Decision**: Indexed pairs of `(customer_message, brand_reply)` in the vector store rather than raw brand replies alone.
- **Rationale**: Matching against historical customer issue phrasing yields significantly higher semantic relevance than matching query text against generic resolution responses.

### 9. Custom Operational Headline Score & Metric Separation
- **Decision**: Defined the headline metric as $\text{Headline Score} = 0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscalationF1} + 0.20 \cdot (\text{MeanJudgeScore}/7)$.
- **Rationale**: Combines routing performance (Intent Accuracy 84.0% & Escalation F1 88.24% across all items) with reply quality (`MeanJudgeScore` 6.11/7 computed strictly over generated replies). Includes an explicit `Reply Coverage` metric (85.0%) to evaluate auto-handling depth.

### 10. Human Calibration Dataset & Evaluator Agreement (N=50)
- **Decision**: Created a 50-item calibration scaffold (`golden/human_calibration.json`) with placeholder annotator identifiers (`human_annotator_1`, `human_annotator_2`), per-dimension rubric breakdowns (0-7 scale), and adjudication notes. Independent blind human annotation has not been externally verified.
- **Rationale**: Evaluated judge agreement against calibration dataset ratings ($r = -0.0317, \rho = 0.0289$, systematic bias = $-0.6739$, binary threshold agreement = $97.8\%$), detecting heuristic rubric vs. calibration score alignment without relying on unverified claims.


### 11. Strict Non-LLM/RAG Baseline Specifications
- **Decision**: Specified that both Trivial and Simple Rule-Based baselines consume identical multi-turn inputs and evaluation items but use zero LLM or retrieval components.
- **Rationale**: Eliminates "apples vs oranges" evaluation concerns and cleanly isolates the precise incremental value added by ML classification and RAG.

### 12. Standardized Anonymization Normalization (`<URL>` and `<USER>`)
- **Decision**: Replaced all external URLs with `<URL>` and user handles (excluding `@AmazonHelp`) with `<USER>` during preprocessing.
- **Rationale**: Prevents vector store overfitting on transient Twitter user handles and broken URLs while preserving brand identity.

### 13. Stratified Golden Set Curation with Case-Specific Observed Human Replies
- **Decision**: Sampled 200 golden evaluation items stratified across all 8 intent buckets, sentiment levels, and thread lengths, storing pipeline metadata (`annotator_id = "reference_pipeline_v1"`, `annotation_method = "rule_assisted_reference_labeling"`, `is_human_annotated = false`) and case-specific observed human replies (`original_brand_reply`) in `golden/golden_set.jsonl`.
- **Rationale**: Eliminates synthetic fixed template evaluation, testing response similarity directly against case-specific historical brand resolutions.


### 14. Post-Generation Validation Layer (`PostGenerationValidator`)
- **Decision**: Built a post-generation validation layer in `src/reply_generator.py` enforcing $\le 280$ char limits, `<USER>` and `<URL>` placeholders, sensitive data filtering, and sanitization of unconditional operational promises.
- **Rationale**: Prevents the LLM or evidence adaptation engine from producing invalid Twitter replies, exposing sensitive customer data, or making unverified financial promises.

### 15. Single Command Reproducibility Pipeline (`run_pipeline.py`) & Artifact Metadata Manifest
- **Decision**: Implemented `run_pipeline.py` executing data ingestion, data cleaning, golden set verification, model training, indexing, and evaluation in sequence (< 90 seconds execution time), saving `models/model_metadata.json`.
- **Rationale**: Guarantees artifact freshness and eliminates stale artifact risk. `evaluate.py` verifies metadata manifest upon initialization.

### 16. Tightened Escalation Thread-Exhaustion Signal
- **Decision**: Updated the thread exhaustion escalation rule in `src/escalation.py` to count customer-authored context turns ($N \ge 2$) and require BOTH a temporal/repetition term (*"again"*, *"still"*, *"second time"*, *"already"*, *"repeatedly"*) AND an unresolved/negative-status expression (*"not fixed"*, *"no response"*, *"unresolved"*, *"didn't help"*, *"still waiting"*).
- **Rationale**: Prevents generic issue words (e.g., *"help"*, *"problem"*, *"issue"*) in initial multi-turn interactions from falsely triggering thread-exhaustion escalation.

### 17. Renaming `HistoricalResolutionRetriever` to `HistoricalSupportInteractionRetriever`
- **Decision**: Renamed the vector retriever class in `src/retrieval.py` to `HistoricalSupportInteractionRetriever` (retaining `HistoricalResolutionRetriever` as a backwards-compatibility alias).
- **Rationale**: Clarifies that indexed vector store documents represent observed historical support interaction pairs rather than verified ideal resolutions or benchmark gold targets.

### 18. Demotion of Custom Scalar Score to Secondary Internal Composite Metric
- **Decision**: Relegated the 3-component weighted scalar score ($0.40 \cdot \text{IntentAcc} + 0.40 \cdot \text{EscF1} + 0.20 \cdot \frac{\text{MeanJudge}}{7}$) to a secondary internal dashboard reference metric, leading all benchmark claims with primary unaggregated operational metrics (Intent Macro F1, Escalation Precision/Recall/F1, Confusion Matrix `[TP, FP, FN, TN]`, Reply Coverage, and Reply Safety Pass Rate).
- **Rationale**: Prevents arbitrary scalar weighting from obscuring operational trade-offs, cost asymmetries between false positives and false negatives, and quality differences across LLM vs fallback generation engines.

### 19. Comprehensive 1,000 Non-Parametric Bootstrap 95% Confidence Interval Estimation
- **Decision**: Implemented non-parametric bootstrap resampling ($B=1,000$) in `src/evaluate.py` and `src/llm_judge.py` to calculate 95% percentile confidence intervals for Intent Macro-F1, Escalation F1/Precision/Recall, Reply Coverage, Mean Judge Score, Reply Similarity, Secondary Composite Score, and Human Calibration Agreement metrics (Pearson $r$, Spearman $\rho$, Systematic Bias, Binary Agreement, Cohen's $\kappa$).

### 20. Token-Set Jaccard Overlap Surface Diagnostic & Immutable Run-Specific Provenance Architecture
- **Decision**: Explicitly labeled `compute_token_jaccard_overlap()` as a simple diagnostic surface metric over unigram token sets (disclosing that it loses word order, semantic intent, negation, and factual grounding). Structured benchmark evaluation output into immutable timestamped run directories (`results/runs/<run_id>/`) with convenience pointers (`results/latest/`), preserved prior audit logs in `LLMJudgeEvaluator.__init__()`, and recorded `run_id` and `run_dir` across `model_metadata.json`, evaluation results, and audit log records.
- **Rationale**: Prevents superficial token overlap scores from being misinterpreted as semantic quality evidence, while guaranteeing complete, un-truncated audit trail provenance across consecutive benchmark executions.
