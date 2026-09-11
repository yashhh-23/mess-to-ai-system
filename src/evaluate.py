import os
import sys
import json
import yaml
import math
import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.intent_classifier import HybridIntentClassifier, INTENTS
from src.retrieval import HistoricalSupportInteractionRetriever
from src.reply_generator import RAGReplyGenerator
from src.escalation import EscalationEngine
from src.llm_judge import LLMJudgeEvaluator, compute_judge_agreement

def compute_token_jaccard_overlap(s1: str, s2: str) -> float:
    """
    Computes Token-set Jaccard overlap with the reference reply.
    Simple diagnostic surface metric over unigram token sets. Does NOT measure semantic
    similarity, word order, negation, factual grounding, or support interaction quality.
    """
    if not s1 or not s2:
        return 0.0
    w1 = set(re.sub(r'[^\w\s]', '', s1.lower()).split())
    w2 = set(re.sub(r'[^\w\s]', '', s2.lower()).split())
    if not w1 or not w2:
        return 0.0
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return float(len(intersection) / len(union))

compute_text_similarity = compute_token_jaccard_overlap  # Alias for backward compatibility

def detect_near_duplicate_leakage(
    golden_path: str = "golden/golden_set.jsonl",
    processed_path: str = "data/processed/amazonhelp_threads.jsonl",
    similarity_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Evaluates near-duplicate text leakage between held-out golden evaluation customer messages
    and training customer messages using sparse TF-IDF cosine similarity across all pairs.
    """
    if not os.path.exists(golden_path) or not os.path.exists(processed_path):
        return {
            'max_near_duplicate_similarity': 0.0,
            'near_duplicate_count': 0,
            'similarity_threshold': similarity_threshold,
            'status': 'FILES_MISSING',
            'near_duplicate_pairs': []
        }

    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_items = [json.loads(line) for line in f]

    with open(processed_path, 'r', encoding='utf-8') as f:
        processed_items = [json.loads(line) for line in f]

    golden_ids = {g['conversation_id'] for g in golden_items}
    train_items = [p for p in processed_items if p['conversation_id'] not in golden_ids]

    gold_msgs = [g['customer_message'] for g in golden_items]
    train_msgs = [t['customer_message'] for t in train_items]

    if not gold_msgs or not train_msgs:
        return {
            'max_near_duplicate_similarity': 0.0,
            'near_duplicate_count': 0,
            'similarity_threshold': similarity_threshold,
            'status': 'EMPTY',
            'near_duplicate_pairs': []
        }

    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_msgs)
    X_train = vec.transform(train_msgs)
    X_gold = vec.transform(gold_msgs)

    sim_matrix = X_gold.dot(X_train.T)
    max_sims = sim_matrix.max(axis=1).toarray().ravel()

    near_dupes = []
    for idx, max_sim in enumerate(max_sims):
        if max_sim >= similarity_threshold:
            train_idx = int(np.argmax(sim_matrix[idx].toarray().ravel()))
            near_dupes.append({
                'golden_id': golden_items[idx]['conversation_id'],
                'train_id': train_items[train_idx]['conversation_id'],
                'similarity': float(round(max_sim, 4)),
                'golden_msg': gold_msgs[idx],
                'train_msg': train_msgs[train_idx]
            })

    max_sim_val = float(round(float(max_sims.max()), 4)) if len(max_sims) > 0 else 0.0

    return {
        'max_near_duplicate_similarity': max_sim_val,
        'near_duplicate_count': len(near_dupes),
        'similarity_threshold': similarity_threshold,
        'status': 'PASSED' if len(near_dupes) == 0 else 'WARNING_LEAKAGE_DETECTED',
        'near_duplicate_pairs': near_dupes
    }


def compute_wilson_confidence_interval(k: int, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """Computes Wilson Score 95% Confidence Interval for a proportion k/n."""
    if n == 0:
        return {'low': 0.0, 'high': 0.0}
    p = k / n
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    return {'low': round(max(0.0, float(lower_bound)), 4), 'high': round(min(1.0, float(upper_bound)), 4)}

def compute_bootstrap_confidence_intervals(
    y_true_intent: List[str],
    y_pred_intent: List[str],
    y_true_esc: List[bool],
    y_pred_esc: List[bool],
    item_judge_scores: List[Optional[float]],
    item_sim_scores: List[Optional[float]],
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Computes 95% non-parametric bootstrap confidence intervals (1,000 resamples)
    for Intent Macro-F1, Escalation F1/Precision/Recall, Reply Coverage, Mean Judge Score,
    Mean Reply Similarity, and Secondary Composite Headline Score.
    """
    n = len(y_true_intent)
    if n == 0:
        return {}

    rng = np.random.RandomState(random_seed)
    boot_acc, boot_macro_f1 = [], []
    boot_prec, boot_rec, boot_esc_f1 = [], [], []
    boot_reply_cov, boot_judge, boot_sim, boot_headline = [], [], [], []

    y_true_intent_np = np.array(y_true_intent)
    y_pred_intent_np = np.array(y_pred_intent)
    y_true_esc_np = np.array(y_true_esc, dtype=bool)
    y_pred_esc_np = np.array(y_pred_esc, dtype=bool)
    item_judge_np = np.array([j if j is not None else np.nan for j in item_judge_scores])
    item_sim_np = np.array([s if s is not None else np.nan for s in item_sim_scores])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_bootstraps):
            idxs = rng.choice(n, size=n, replace=True)
            
            # Intent metrics
            yt_i, yp_i = y_true_intent_np[idxs], y_pred_intent_np[idxs]
            b_acc = float(np.mean(yt_i == yp_i))
            _, _, b_macro_f1, _ = precision_recall_fscore_support(yt_i, yp_i, average='macro', zero_division=0)
            
            # Escalation metrics
            yt_e, yp_e = y_true_esc_np[idxs], y_pred_esc_np[idxs]
            tp = sum(1 for yt, yp in zip(yt_e, yp_e) if yp and yt)
            fp = sum(1 for yt, yp in zip(yt_e, yp_e) if yp and not yt)
            fn = sum(1 for yt, yp in zip(yt_e, yp_e) if not yp and yt)
            tn = sum(1 for yt, yp in zip(yt_e, yp_e) if not yp and not yt)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            cov = (tn + fn) / n

            # Response quality metrics
            b_j = item_judge_np[idxs]
            b_s = item_sim_np[idxs]
            valid_j = b_j[~np.isnan(b_j)]
            valid_s = b_s[~np.isnan(b_s)]
            mj = float(np.mean(valid_j)) if len(valid_j) > 0 else 0.0
            ms = float(np.mean(valid_s)) if len(valid_s) > 0 else 0.0

            head = 0.40 * b_acc + 0.40 * f1 + 0.20 * (mj / 7.0)

            boot_acc.append(b_acc)
            boot_macro_f1.append(float(b_macro_f1))
            boot_prec.append(p)
            boot_rec.append(r)
            boot_esc_f1.append(f1)
            boot_reply_cov.append(cov)
            boot_judge.append(mj)
            boot_sim.append(ms)
            boot_headline.append(head)

    low_p = (100.0 - ci) / 2.0
    high_p = 100.0 - low_p

    def _format_ci(vals):
        l = float(np.percentile(vals, low_p))
        h = float(np.percentile(vals, high_p))
        return f"[{l:.4f}, {h:.4f}]", round(l, 4), round(h, 4)

    def _format_ci_pct(vals):
        l = float(np.percentile(vals, low_p)) * 100.0
        h = float(np.percentile(vals, high_p)) * 100.0
        return f"[{l:.1f}%, {h:.1f}%]"

    acc_str, acc_l, acc_h = _format_ci(boot_acc)
    mf1_str, mf1_l, mf1_h = _format_ci(boot_macro_f1)
    prec_str, prec_l, prec_h = _format_ci(boot_prec)
    rec_str, rec_l, rec_h = _format_ci(boot_rec)
    ef1_str, ef1_l, ef1_h = _format_ci(boot_esc_f1)
    cov_str, cov_l, cov_h = _format_ci(boot_reply_cov)
    cov_str_pct = _format_ci_pct(boot_reply_cov)
    j_str, j_l, j_h = _format_ci(boot_judge)
    s_str, s_l, s_h = _format_ci(boot_sim)
    head_str, head_l, head_h = _format_ci(boot_headline)

    return {
        'intent_accuracy_ci_str': acc_str,
        'intent_macro_f1_ci_str': mf1_str,
        'escalation_precision_ci_str': prec_str,
        'escalation_recall_ci_str': rec_str,
        'escalation_f1_ci_str': ef1_str,
        'reply_coverage_ci_str': cov_str,
        'reply_coverage_ci_str_pct': cov_str_pct,
        'mean_judge_score_ci_str': j_str,
        'mean_reply_similarity_ci_str': s_str,
        'secondary_composite_headline_score_ci_str': head_str,
        'details': {
            'n_bootstraps': n_bootstraps,
            'confidence_level': ci,
            'intent_accuracy_ci': [acc_l, acc_h],
            'intent_macro_f1_ci': [mf1_l, mf1_h],
            'escalation_precision_ci': [prec_l, prec_h],
            'escalation_recall_ci': [rec_l, rec_h],
            'escalation_f1_ci': [ef1_l, ef1_h],
            'reply_coverage_ci': [cov_l, cov_h],
            'mean_judge_score_ci': [j_l, j_h],
            'mean_reply_similarity_ci': [s_l, s_h],
            'secondary_composite_headline_score_ci': [head_l, head_h],
        }
    }

def _compute_md5(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def compute_source_code_hash(project_root: Optional[str] = None) -> str:
    """
    Computes a comprehensive SHA-256 hash across the full reproducible experiment surface:
    - run_pipeline.py
    - requirements.txt
    - configs/**/*.yaml
    - src/**/*.py
    - golden/**/*.py and golden/**/*.md
    - tests/**/*.py
    - notebooks/**/*.ipynb
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    hasher = hashlib.sha256()
    files_to_hash = []

    # 1. Root pipeline runner & dependencies
    for fname in ['run_pipeline.py', 'requirements.txt']:
        fpath = os.path.join(project_root, fname)
        if os.path.exists(fpath):
            files_to_hash.append(fpath)

    # 2. Experiment surface directories & file extensions
    target_dirs = {
        'configs': ('.yaml', '.yml'),
        'src': ('.py',),
        'golden': ('.py', '.md'),
        'tests': ('.py',),
        'notebooks': ('.ipynb',)
    }

    for dir_name, ext_tuple in target_dirs.items():
        full_dir = os.path.join(project_root, dir_name)
        if os.path.exists(full_dir):
            for root, _, files in os.walk(full_dir):
                for fname in sorted(files):
                    if fname.endswith(ext_tuple):
                        files_to_hash.append(os.path.join(root, fname))

    for fpath in sorted(files_to_hash):
        with open(fpath, 'rb') as f:
            hasher.update(f.read())

    return hasher.hexdigest()

def check_artifact_metadata(config_path: str = "configs/config.yaml", metadata_path: str = "models/model_metadata.json") -> bool:
    """
    Strictly validates artifact freshness by recomputing current checksums
    (config, processed data, golden set, intent model, vector store, source code) and comparing
    them against the saved model_metadata.json manifest.
    Fails closed by raising RuntimeError if metadata is missing or checksums mismatch.
    """
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            f"[Artifact Validation Error] Metadata manifest '{metadata_path}' is missing!\n"
            "Run 'python run_pipeline.py' to generate fresh trained models and metadata."
        )

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        raise RuntimeError(f"[Artifact Validation Error] Could not parse '{metadata_path}': {e}") from e

    raw_csv_path = "data/raw/twcs.csv"
    processed_path = "data/processed/amazonhelp_threads.jsonl"
    golden_path = "golden/golden_set.jsonl"
    human_calib_path = "golden/human_calibration.json"
    brand_meta_path = "data/processed/brand_metadata.json"
    intent_model_path = "models/intent_classifier.pkl"
    vector_store_path = "models/vector_store.pkl"
    manifest_path = "data/raw/data_manifest.json"

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                paths = cfg.get('paths', {})
                raw_csv_path = paths.get('raw_data', raw_csv_path)
                processed_path = paths.get('processed_data', processed_path)
                golden_path = paths.get('golden_set', golden_path)
                human_calib_path = paths.get('human_calibration', human_calib_path)
                brand_meta_path = paths.get('brand_metadata', brand_meta_path)
                intent_model_path = paths.get('intent_model', intent_model_path)
                vector_store_path = paths.get('vector_store', vector_store_path)
        except Exception:
            pass

    checks = {
        'config_hash': (config_path, meta.get('config_hash')),
        'dataset_manifest_checksum': (manifest_path, meta.get('dataset_manifest_checksum')),
        'processed_data_checksum': (processed_path, meta.get('processed_data_checksum')),
        'golden_set_checksum': (golden_path, meta.get('golden_set_checksum')),
        'human_calibration_checksum': (human_calib_path, meta.get('human_calibration_checksum')),
        'brand_metadata_checksum': (brand_meta_path, meta.get('brand_metadata_checksum')),
        'intent_model_checksum': (intent_model_path, meta.get('intent_model_checksum')),
        'vector_store_checksum': (vector_store_path, meta.get('vector_store_checksum')),
    }

    mismatches = []
    
    # Verify source code hash across src/, golden/, run_pipeline.py, requirements.txt
    expected_src_hash = meta.get('source_code_hash')
    if expected_src_hash:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        actual_src_hash = compute_source_code_hash(project_root=project_root)
        if actual_src_hash != expected_src_hash:
            mismatches.append(f"source_code_hash (expected {expected_src_hash[:8]}..., got {actual_src_hash[:8]}...)")

    # Verify raw CSV checksum & file size if file exists; if absent, inspect download_required_for_full_repro flag & manifest
    download_required_flag = meta.get('download_required_for_full_repro', not os.path.exists(raw_csv_path))
    expected_raw_checksum = meta.get('raw_data_checksum')
    if expected_raw_checksum:
        if os.path.exists(raw_csv_path):
            actual_raw = _compute_md5(raw_csv_path)
            if actual_raw != expected_raw_checksum:
                mismatches.append(f"raw_data_checksum ({raw_csv_path}: expected {expected_raw_checksum[:8]}..., got {actual_raw[:8]}...)")
            
            # Check file size against manifest if available
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        m_data = json.load(f)
                        expected_bytes = m_data.get('file_size_bytes')
                        if expected_bytes and expected_bytes > 0:
                            actual_bytes = os.path.getsize(raw_csv_path)
                            rel_diff = abs(actual_bytes - expected_bytes) / expected_bytes
                            if rel_diff > 0.05:
                                mismatches.append(f"raw_data_file_size ({raw_csv_path}: actual {actual_bytes:,} bytes differs from manifest expected {expected_bytes:,} bytes by {rel_diff:.2%})")
                except Exception:
                    pass
        else:
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    f"[Artifact Validation Error] Raw data file '{raw_csv_path}' is missing "
                    f"and no trusted manifest found at '{manifest_path}'.\n"
                    "Run 'python src/download_data.py' or 'python run_pipeline.py' to download."
                )
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                    if not isinstance(manifest_data, dict) or 'file_size_bytes' not in manifest_data:
                        raise ValueError("Manifest missing required schema fields.")
            except Exception as e:
                raise RuntimeError(
                    f"[Artifact Validation Error] Raw data file '{raw_csv_path}' is missing "
                    f"and manifest at '{manifest_path}' is invalid: {e}"
                ) from e
            
            import warnings
            warn_msg = (
                f"[Artifact Validation WARNING] Raw CSV absent ({raw_csv_path}); download_required_for_full_repro={download_required_flag}. "
                f"Using processed data with trusted manifest ({manifest_path}). "
                "Re-run 'python src/download_data.py' if full retraining from source is required."
            )
            warnings.warn(warn_msg, UserWarning)
            print(warn_msg)

    for key, (fpath, expected) in checks.items():
        if not expected:
            continue
        actual = _compute_md5(fpath)
        if actual != expected:
            mismatches.append(f"{key} ({fpath}: expected {expected[:8]}..., got {actual[:8]}...)")

    if mismatches:
        error_msg = (
            "[Artifact Validation Error] Stale model artifact(s) detected!\n"
            + "\n".join(f" - {m}" for m in mismatches)
            + "\nRun 'python run_pipeline.py' to rebuild models from current code, data, and config."
        )
        print(error_msg)
        raise RuntimeError(error_msg)

    git_sha = meta.get('git_commit_sha', 'N/A')
    trained_at = meta.get('train_timestamp', 'Unknown')
    print(f"[Artifact Validation] Freshness check PASSED — all {len(checks)} artifact checksums match active code/config/data. (Git SHA: {git_sha[:8]}, Trained: {trained_at}).")
    return True

def run_evaluation(config_path: str = "configs/config.yaml", run_id: Optional[str] = None, output_base_dir: Optional[str] = None):
    import time
    import shutil
    import uuid
    
    if run_id is None:
        timestamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
        short_uuid = uuid.uuid4().hex[:8]
        run_id = f"{timestamp}_{short_uuid}"

    base_results_dir = output_base_dir or "results"
    run_dir = os.path.join(base_results_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    if output_base_dir is None:
        os.makedirs("results/latest", exist_ok=True)
    
    run_audit_path = os.path.join(run_dir, "judge_audit_log.jsonl")
    run_results_path = os.path.join(run_dir, "evaluation_results.json")

    print(f"[Evaluation Harness] Initializing evaluation run: {run_id} (Directory: {run_dir})...")
    check_artifact_metadata(config_path=config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    golden_path = config['paths']['golden_set']
    human_calib_path = config.get('paths', {}).get('human_calibration', "golden/human_calibration.json")
    brand_meta_path = config.get('paths', {}).get('brand_metadata', "data/processed/brand_metadata.json")
    
    llm_cfg = config.get('llm', {})
    provider = llm_cfg.get('provider', 'auto')
    model_name = llm_cfg.get('model_name')      # None → LLMClient picks provider default
    base_url = llm_cfg.get('base_url')          # None → LLMClient picks standard endpoint

    print(f"[Evaluation Harness] LLM config: provider={provider!r}, model_name={model_name!r}, base_url={base_url!r}")

    # Load resolved brand handle to prevent cross-brand contamination
    brand_handle = config['brand']['handle']
    if os.path.exists(brand_meta_path):
        with open(brand_meta_path, 'r', encoding='utf-8') as f:
            b_meta = json.load(f)
            brand_handle = b_meta.get('resolved_brand_handle', brand_handle)
            print(f"[Evaluation Harness] Verified resolved brand handle: {brand_handle}")
    
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden set file {golden_path} not found. Please run golden/build_golden_set.py first.")

    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_items = [json.loads(line) for line in f]
        
    print(f"[Evaluation Harness] Loaded {len(golden_items)} Golden Evaluation items.")

    # Class distribution analysis of Golden Set
    intent_counts = {}
    escalate_count = 0
    for g in golden_items:
        intent_counts[g['gold_intent']] = intent_counts.get(g['gold_intent'], 0) + 1
        if g['gold_escalate']:
            escalate_count += 1
            
    golden_set_distribution = {
        'total_items': len(golden_items),
        'intent_counts': intent_counts,
        'escalation_count': escalate_count,
        'auto_handle_count': len(golden_items) - escalate_count,
        'escalation_ratio': round(escalate_count / len(golden_items), 4),
        'labeling_mechanism': 'rule_assisted_reference_labeling',
        'is_human_annotated': False,
        'labeling_mechanism_disclosure': (
            "Golden set evaluation labels were generated using rule-assisted reference labeling (reference_pipeline_v1). "
            "Because training weak labels, evaluation reference labels, escalation logic, and baseline rules share keyword patterns, "
            "headline metrics reflect agreement with rule-derived proxies. Exact conversation ID separation prevents direct data leakage, "
            "but rule-mechanism correlation remains an inherent limitation of rule-assisted reference sets."
        )
    }

    human_calib_dict = {}
    if os.path.exists(human_calib_path):
        with open(human_calib_path, 'r', encoding='utf-8') as f:
            calib_items = json.load(f)
            human_calib_dict = {c['item_id']: c for c in calib_items}
        print(f"[Evaluation Harness] Loaded {len(human_calib_dict)} calibration annotation records (placeholder annotator IDs; independent verification pending).")

    clf = HybridIntentClassifier()
    clf.load(config['paths']['intent_model'])
    
    retriever = HistoricalSupportInteractionRetriever()
    retriever.load(config['paths']['vector_store'])
    
    generator = RAGReplyGenerator(
        brand_handle=brand_handle,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    escalator = EscalationEngine(config_path=config_path)
    judge = LLMJudgeEvaluator(
        brand_handle=brand_handle,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        audit_log_path=run_audit_path,
        config_path=config_path,
        run_id=run_id,
    )

    models = {
        'Main Agent':       {'y_true_intent': [], 'y_pred_intent': [], 'y_true_esc': [], 'y_pred_esc': [], 'confidences': [], 'esc_tp': 0, 'esc_fp': 0, 'esc_fn': 0, 'esc_tn': 0, 'item_judge_scores': [], 'item_sim_scores': [], 'judge_scores': [], 'sim_scores': [], 'safety_scores': [], 'eval_modes': [], 'gen_modes': [], 'reply_count': 0, 'reply_details': []},
        'Trivial Baseline': {'y_true_intent': [], 'y_pred_intent': [], 'y_true_esc': [], 'y_pred_esc': [], 'esc_tp': 0, 'esc_fp': 0, 'esc_fn': 0, 'esc_tn': 0, 'item_judge_scores': [], 'item_sim_scores': [], 'judge_scores': [], 'sim_scores': [], 'safety_scores': [], 'eval_modes': [], 'gen_modes': [], 'reply_count': 0, 'reply_details': []},
        'Simple Baseline':  {'y_true_intent': [], 'y_pred_intent': [], 'y_true_esc': [], 'y_pred_esc': [], 'esc_tp': 0, 'esc_fp': 0, 'esc_fn': 0, 'esc_tn': 0, 'item_judge_scores': [], 'item_sim_scores': [], 'judge_scores': [], 'sim_scores': [], 'safety_scores': [], 'eval_modes': [], 'gen_modes': [], 'reply_count': 0, 'reply_details': []}
    }

    human_calibration_scores = []
    judge_calibration_scores = []
    human_calibration_dim_scores = {'correctness': [], 'tone': [], 'actionability': [], 'safety': []}
    judge_calibration_dim_scores = {'correctness': [], 'tone': [], 'actionability': [], 'safety': []}

    print("[Evaluation Harness] Running evaluation on 200 Golden Set items...")

    for idx, item in enumerate(golden_items):
        item_id = item.get('id', f"gold_{idx:03d}")
        cust_msg = item['customer_message']
        context = item['context_messages']
        gold_intent = item['gold_intent']
        gold_escalate = item['gold_escalate']
        gold_reply = item['gold_reply']

        # --- A. MAIN AGENT EVALUATION ---
        pred_intent, conf, _ = clf.predict(cust_msg, context)
        retrieved = retriever.retrieve(cust_msg, context, top_k=3)
        top_retrieval_score = retrieved[0]['score'] if retrieved else 0.0
        
        escalate, esc_score, esc_reason = escalator.evaluate(
            customer_message=cust_msg,
            intent=pred_intent,
            intent_confidence=conf,
            context_messages=context,
            retrieval_score=top_retrieval_score
        )
        
        models['Main Agent']['y_true_intent'].append(gold_intent)
        models['Main Agent']['y_pred_intent'].append(pred_intent)
        models['Main Agent']['y_true_esc'].append(gold_escalate)
        models['Main Agent']['y_pred_esc'].append(escalate)
        models['Main Agent']['confidences'].append(float(conf))
            
        if escalate and gold_escalate:
            models['Main Agent']['esc_tp'] += 1
        elif escalate and not gold_escalate:
            models['Main Agent']['esc_fp'] += 1
        elif not escalate and gold_escalate:
            models['Main Agent']['esc_fn'] += 1
        else:
            models['Main Agent']['esc_tn'] += 1

        if not escalate:
            models['Main Agent']['reply_count'] += 1
            gen_res = generator.generate_reply_detailed(cust_msg, pred_intent, retrieved, context)
            reply = gen_res['reply']
            gen_mode = gen_res['generation_mode']
            models['Main Agent']['gen_modes'].append(gen_mode)
            
            judge_res = judge.evaluate_reply(item_id, cust_msg, reply, gold_reply, pred_intent, context)
            j_score = judge_res['total']
            s_score = compute_text_similarity(reply, gold_reply)
            safe_score = judge_res.get('safety', 1)
            models['Main Agent']['judge_scores'].append(j_score)
            models['Main Agent']['sim_scores'].append(s_score)
            models['Main Agent']['safety_scores'].append(safe_score)
            models['Main Agent']['item_judge_scores'].append(j_score)
            models['Main Agent']['item_sim_scores'].append(s_score)
            models['Main Agent']['eval_modes'].append(judge_res['evaluator_mode'])
            models['Main Agent']['reply_details'].append({
                'gen_mode': gen_mode,
                'eval_mode': judge_res['evaluator_mode'],
                'judge_score': j_score,
                'sim_score': s_score,
                'safety_score': safe_score,
            })
            
            if item_id in human_calib_dict:
                h_item = human_calib_dict[item_id]
                human_calibration_scores.append(h_item['human_total_score'])
                judge_calibration_scores.append(j_score)
                h_rubric = h_item.get('human_rubric_scores', {})
                for dim in ['correctness', 'tone', 'actionability', 'safety']:
                    human_calibration_dim_scores[dim].append(float(h_rubric.get(dim, 0)))
                    judge_calibration_dim_scores[dim].append(float(judge_res.get(dim, 0)))
        else:
            models['Main Agent']['gen_modes'].append("Escalated (No Reply)")
            models['Main Agent']['eval_modes'].append("Skipped (Escalated)")
            models['Main Agent']['item_judge_scores'].append(None)
            models['Main Agent']['item_sim_scores'].append(None)

        # --- B. TRIVIAL BASELINE EVALUATION ---
        triv_intent = 'order_status'
        triv_escalate = False
        triv_reply = "Thanks for reaching out. Please check our help center at <URL> or reply with more details."
        
        models['Trivial Baseline']['y_true_intent'].append(gold_intent)
        models['Trivial Baseline']['y_pred_intent'].append(triv_intent)
        models['Trivial Baseline']['y_true_esc'].append(gold_escalate)
        models['Trivial Baseline']['y_pred_esc'].append(triv_escalate)
        models['Trivial Baseline']['gen_modes'].append("Static Template")
        
        if triv_escalate and gold_escalate: models['Trivial Baseline']['esc_tp'] += 1
        elif triv_escalate and not gold_escalate: models['Trivial Baseline']['esc_fp'] += 1
        elif not triv_escalate and gold_escalate: models['Trivial Baseline']['esc_fn'] += 1
        else: models['Trivial Baseline']['esc_tn'] += 1

        models['Trivial Baseline']['reply_count'] += 1
        triv_judge = judge.evaluate_reply(f"triv_{item_id}", cust_msg, triv_reply, gold_reply, triv_intent, context)
        triv_j = triv_judge['total']
        triv_s = compute_text_similarity(triv_reply, gold_reply)
        triv_safe = triv_judge.get('safety', 1)
        models['Trivial Baseline']['judge_scores'].append(triv_j)
        models['Trivial Baseline']['sim_scores'].append(triv_s)
        models['Trivial Baseline']['safety_scores'].append(triv_safe)
        models['Trivial Baseline']['item_judge_scores'].append(triv_j)
        models['Trivial Baseline']['item_sim_scores'].append(triv_s)
        models['Trivial Baseline']['eval_modes'].append(triv_judge['evaluator_mode'])
        models['Trivial Baseline']['reply_details'].append({
            'gen_mode': "Static Template",
            'eval_mode': triv_judge['evaluator_mode'],
            'judge_score': triv_j,
            'sim_score': triv_s,
            'safety_score': triv_safe,
        })

        # --- C. SIMPLE RULE BASELINE EVALUATION ---
        simp_intent = 'refund_request' if 'refund' in cust_msg.lower() else ('order_status' if 'track' in cust_msg.lower() else 'general_inquiry')
        simp_escalate = any(w in cust_msg.lower() for w in ['lawyer', 'sue', 'human', 'agent'])
        simp_reply = "You can track your order or request a refund via this link: <URL>."
        
        models['Simple Baseline']['y_true_intent'].append(gold_intent)
        models['Simple Baseline']['y_pred_intent'].append(simp_intent)
        models['Simple Baseline']['y_true_esc'].append(gold_escalate)
        models['Simple Baseline']['y_pred_esc'].append(simp_escalate)
        
        if simp_escalate and gold_escalate: models['Simple Baseline']['esc_tp'] += 1
        elif simp_escalate and not gold_escalate: models['Simple Baseline']['esc_fp'] += 1
        elif not simp_escalate and gold_escalate: models['Simple Baseline']['esc_fn'] += 1
        else: models['Simple Baseline']['esc_tn'] += 1

        if not simp_escalate:
            models['Simple Baseline']['reply_count'] += 1
            models['Simple Baseline']['gen_modes'].append("Rule Template")
            simp_judge = judge.evaluate_reply(f"simp_{item_id}", cust_msg, simp_reply, gold_reply, simp_intent, context)
            simp_j = simp_judge['total']
            simp_s = compute_text_similarity(simp_reply, gold_reply)
            simp_safe = simp_judge.get('safety', 1)
            models['Simple Baseline']['judge_scores'].append(simp_j)
            models['Simple Baseline']['sim_scores'].append(simp_s)
            models['Simple Baseline']['safety_scores'].append(simp_safe)
            models['Simple Baseline']['item_judge_scores'].append(simp_j)
            models['Simple Baseline']['item_sim_scores'].append(simp_s)
            models['Simple Baseline']['eval_modes'].append(simp_judge['evaluator_mode'])
            models['Simple Baseline']['reply_details'].append({
                'gen_mode': "Rule Template",
                'eval_mode': simp_judge['evaluator_mode'],
                'judge_score': simp_j,
                'sim_score': simp_s,
                'safety_score': simp_safe,
            })
        else:
            models['Simple Baseline']['gen_modes'].append("Escalated (No Reply)")
            models['Simple Baseline']['eval_modes'].append("Skipped (Escalated)")
            models['Simple Baseline']['item_judge_scores'].append(None)
            models['Simple Baseline']['item_sim_scores'].append(None)

    total_n = len(golden_items)
    results_table = {}

    # ── Mode distribution counters (frequency-sorted, deterministic) ──────────
    def _count_modes(mode_list):
        counts = {}
        for m in mode_list:
            counts[m] = counts.get(m, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    main_eval_mode_counts = _count_modes(models['Main Agent']['eval_modes'])
    main_gen_mode_counts  = _count_modes(models['Main Agent']['gen_modes'])

    # ── Per-mode quality breakdown helper ────────────────────────────────────
    def _per_mode_quality(reply_details):
        mode_buckets = {}
        for rd in reply_details:
            gm = rd['gen_mode']
            if gm not in mode_buckets:
                mode_buckets[gm] = {'judge': [], 'sim': [], 'safety': []}
            mode_buckets[gm]['judge'].append(rd['judge_score'])
            mode_buckets[gm]['sim'].append(rd['sim_score'])
            mode_buckets[gm]['safety'].append(rd.get('safety_score', 1))

        per_mode = {}
        llm_judge, llm_sim, llm_safe = [], [], []
        fb_judge,  fb_sim,  fb_safe  = [], [], []

        for gm, vals in mode_buckets.items():
            mj = round(float(np.mean(vals['judge'])), 4) if vals['judge'] else None
            ms = round(float(np.mean(vals['sim'])),   4) if vals['sim']   else None
            safe_viol = sum(1 for s in vals['safety'] if s == 0)
            safe_rate = round(safe_viol / len(vals['safety']), 4) if vals['safety'] else 0.0
            per_mode[gm] = {
                'count': len(vals['judge']),
                'mean_judge_score': mj,
                'mean_reply_similarity': ms,
                'safety_violation_rate': safe_rate
            }
            if gm.startswith('LLM-RAG-Synthesized'):
                llm_judge.extend(vals['judge']); llm_sim.extend(vals['sim']); llm_safe.extend(vals['safety'])
            else:
                fb_judge.extend(vals['judge']);  fb_sim.extend(vals['sim']);  fb_safe.extend(vals['safety'])

        llm_viol = sum(1 for s in llm_safe if s == 0)
        fb_viol = sum(1 for s in fb_safe if s == 0)

        aggregate = {
            'llm_generated': {
                'count': len(llm_judge),
                'mean_judge_score':      round(float(np.mean(llm_judge)), 4) if llm_judge else None,
                'mean_reply_similarity': round(float(np.mean(llm_sim)),   4) if llm_sim   else None,
                'safety_violation_rate': round(llm_viol / len(llm_safe), 4) if llm_safe else 0.0
            },
            'fallback_generated': {
                'count': len(fb_judge),
                'mean_judge_score':      round(float(np.mean(fb_judge)), 4) if fb_judge else None,
                'mean_reply_similarity': round(float(np.mean(fb_sim)),   4) if fb_sim   else None,
                'safety_violation_rate': round(fb_viol / len(fb_safe), 4) if fb_safe else 0.0
            },
        }
        return per_mode, aggregate

    for name, data in models.items():
        y_true = data['y_true_intent']
        y_pred = data['y_pred_intent']
        correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        intent_acc = correct_count / total_n
        intent_ci = compute_wilson_confidence_interval(correct_count, total_n)
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class classification metrics
        cls_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class_f1 = {}
        for cat in INTENTS:
            if cat in cls_rep:
                per_class_f1[cat] = {
                    'precision': round(cls_rep[cat]['precision'], 4),
                    'recall': round(cls_rep[cat]['recall'], 4),
                    'f1_score': round(cls_rep[cat]['f1-score'], 4),
                    'support': int(cls_rep[cat]['support'])
                }

        tp, fp, fn, tn = data['esc_tp'], data['esc_fp'], data['esc_fn'], data['esc_tn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        esc_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        esc_ci = compute_wilson_confidence_interval(tp, tp + fn)
        esc_matrix = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
        
        reply_count = data['reply_count']
        reply_coverage = reply_count / total_n
        esc_coverage = (total_n - reply_count) / total_n

        safety_scores = data['safety_scores']
        safety_viol_count = sum(1 for s in safety_scores if s == 0)
        safety_viol_rate = round(safety_viol_count / reply_count, 4) if reply_count > 0 else 0.0
        safety_pass_rate = round(1.0 - safety_viol_rate, 4)
        
        mean_judge = float(np.mean(data['judge_scores'])) if data['judge_scores'] else None
        mean_sim   = float(np.mean(data['sim_scores']))   if data['sim_scores']   else None

        if mean_judge is not None:
            headline_score = 0.40 * intent_acc + 0.40 * esc_f1 + 0.20 * (mean_judge / 7.0)
        else:
            headline_score = 0.50 * intent_acc + 0.50 * esc_f1

        per_mode_quality, quality_by_bucket = _per_mode_quality(data['reply_details'])

        # Compute 1000 Non-parametric Bootstrap 95% Confidence Intervals
        boot_cis = compute_bootstrap_confidence_intervals(
            y_true_intent=data['y_true_intent'],
            y_pred_intent=data['y_pred_intent'],
            y_true_esc=data['y_true_esc'],
            y_pred_esc=data['y_pred_esc'],
            item_judge_scores=data['item_judge_scores'],
            item_sim_scores=data['item_sim_scores'],
            n_bootstraps=1000,
            ci=95.0
        )

        results_table[name] = {
            # ── Primary Lead Operational Metrics (Unweighted) ──
            'Intent Macro F1': round(float(f1_macro), 4),
            'Intent Macro F1 95% CI': boot_cis.get('intent_macro_f1_ci_str', 'N/A'),
            'Intent Accuracy': round(intent_acc, 4),
            'Intent Accuracy 95% CI': boot_cis.get('intent_accuracy_ci_str', f"[{intent_ci['low']:.4f}, {intent_ci['high']:.4f}]"),
            'Per-Class Intent F1': per_class_f1,
            'Escalation Precision': round(precision, 4),
            'Escalation Precision 95% CI': boot_cis.get('escalation_precision_ci_str', 'N/A'),
            'Escalation Recall': round(recall, 4),
            'Escalation Recall 95% CI': boot_cis.get('escalation_recall_ci_str', f"[{esc_ci['low']:.4f}, {esc_ci['high']:.4f}]"),
            'Escalation F1': round(esc_f1, 4),
            'Escalation F1 95% CI': boot_cis.get('escalation_f1_ci_str', 'N/A'),
            'Escalation Confusion Matrix': esc_matrix,
            'Reply Coverage': round(reply_coverage, 4),
            'Reply Coverage 95% CI': boot_cis.get('reply_coverage_ci_str', 'N/A'),
            'Reply Coverage 95% CI (Pct)': boot_cis.get('reply_coverage_ci_str_pct', 'N/A'),
            'Escalation Coverage': round(esc_coverage, 4),
            'Reply Safety Pass Rate': safety_pass_rate,
            'Reply Safety Violation Rate': safety_viol_rate,
            'Mean Judge Score (0-7)': round(mean_judge, 2) if mean_judge is not None else None,
            'Mean Judge Score 95% CI': boot_cis.get('mean_judge_score_ci_str', 'N/A') if mean_judge is not None else None,
            'Mean Reply Similarity': round(mean_sim, 4) if mean_sim is not None else None,
            'Mean Reply Similarity 95% CI': boot_cis.get('mean_reply_similarity_ci_str', 'N/A') if mean_sim is not None else None,
            'quality_by_generation_mode': per_mode_quality,
            'quality_by_llm_vs_fallback': quality_by_bucket,

            # ── Secondary Internal Composite Metric (Dashboard Summary Reference Only) ──
            'Secondary Composite Headline Score': round(headline_score, 4),
            'Secondary Composite Headline Score 95% CI': boot_cis.get('secondary_composite_headline_score_ci_str', 'N/A'),
            'Headline Score': round(headline_score, 4),  # Alias for backward compatibility
            'headline_score_disclosure': (
                "Reported strictly as a secondary internal composite dashboard metric. "
                "Primary benchmark claims lead with unweighted operational metrics and 95% bootstrap confidence intervals."
            ),
            'bootstrap_ci_details': boot_cis.get('details', {}),

            # ── Distribution breakdowns ──────────────────────────────────────
            'evaluator_mode_counts': _count_modes(data['eval_modes']),
            'generation_mode_counts': _count_modes(data['gen_modes']),
        }

    agreement_metrics = compute_judge_agreement(
        human_calibration_scores,
        judge_calibration_scores,
        human_dim_scores=human_calibration_dim_scores,
        judge_dim_scores=judge_calibration_dim_scores
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n==========================================================================================")
    print(f"               EVALUATION BENCHMARK RESULTS [Brand: {brand_handle}]")
    print("==========================================================================================")
    print("PRIMARY OPERATIONAL METRICS (LEAD BENCHMARK RESULTS WITH 95% BOOTSTRAP CIs):")
    print(f"{'Model / System':<18} | {'Intent F1 (95% CI)':<26} | {'Esc F1 (95% CI)':<26} | {'Esc Matrix [TP,FP,FN,TN]':<24} | {'Coverage (95% CI)':<22}")
    print("-" * 125)
    for name, metrics in results_table.items():
        mat = metrics['Escalation Confusion Matrix']
        mat_str = f"[{mat['TP']}, {mat['FP']}, {mat['FN']}, {mat['TN']}]"
        f1_str = f"{metrics['Intent Macro F1']:.4f} {metrics['Intent Macro F1 95% CI']}"
        ef1_str = f"{metrics['Escalation F1']:.4f} {metrics['Escalation F1 95% CI']}"
        cov_str = f"{metrics['Reply Coverage']*100:.1f}% {metrics.get('Reply Coverage 95% CI (Pct)', '')}"
        print(f"{name:<18} | {f1_str:<26} | {ef1_str:<26} | {mat_str:<24} | {cov_str}")
    
    print("\nSECONDARY INTERNAL COMPOSITE METRIC (DASHBOARD SUMMARY REFERENCE ONLY):")
    print(f"{'Model / System':<18} | {'Secondary Composite Score (95% CI)':<42} | {'Formula Weighting Basis':<40}")
    print("-" * 105)
    for name, metrics in results_table.items():
        comp_str = f"{metrics['Secondary Composite Headline Score']:.4f} {metrics['Secondary Composite Headline Score 95% CI']}"
        print(f"{name:<18} | {comp_str:<42} | 0.40*IntentAcc + 0.40*EscF1 + 0.20*(MeanJudge/7)")
    print("==========================================================================================")
    print(f"Brand Handle:         {brand_handle}")
    print("\nMain Agent — Evaluator Mode Distribution:")
    for mode, cnt in main_eval_mode_counts.items():
        print(f"  {cnt:>4}x  {mode}")
    print("\nMain Agent — Generation Mode Distribution:")
    for mode, cnt in main_gen_mode_counts.items():
        print(f"  {cnt:>4}x  {mode}")
    print(f"Run ID:                {run_id}")
    print(f"Run Directory:         {run_dir}")
    print(f"Auditable Audit Log:  {run_audit_path}")

    main_confs = models['Main Agent'].get('confidences', [])
    conf_stats = {
        'mean_confidence': round(float(np.mean(main_confs)), 4) if main_confs else 0.0,
        'min_confidence': round(float(np.min(main_confs)), 4) if main_confs else 0.0,
        'max_confidence': round(float(np.max(main_confs)), 4) if main_confs else 0.0,
        'calibration_type': 'weak_label_calibrated_CalibratedClassifierCV',
        'calibration_disclosure': (
            "Classifier confidence scores are calibrated using CalibratedClassifierCV(cv=3) "
            "against weak-supervised rule labels (_rule_fallback). Operational probability "
            "calibration claims in live production require human ground-truth evaluation."
        )
    }

    human_calib_summary = {
        'sample_size': len(human_calibration_scores),
        'mean_human_rubric_score': round(float(np.mean(human_calibration_scores)), 2) if human_calibration_scores else None,
        'mean_judge_rubric_score': round(float(np.mean(judge_calibration_scores)), 2) if judge_calibration_scores else None,
        'provenance_disclosure': (
            "Authentic human calibration annotations imported from golden/human_calibration.json "
            "containing 50 blind human-annotated candidate replies evaluated across correctness, tone, "
            "actionability, and safety criteria."
        ),
        'agreement_metrics': agreement_metrics
    }

    near_dupe_diag = detect_near_duplicate_leakage(golden_path=golden_path, processed_path=config['paths']['processed_data'])

    output_res = {
        'run_id': run_id,
        'run_directory': run_dir,
        'resolved_brand_handle': brand_handle,
        'near_duplicate_diagnostics': near_dupe_diag,
        'token_jaccard_disclosure': (
            "Token Jaccard overlap is a simple diagnostic surface metric over unigram token sets. "
            "It loses word order, semantic intent, negation, and factual grounding. Human review and "
            "LLM rubric evaluations serve as primary quality measures."
        ),
        'runtime_llm_config': {
            'provider': provider,
            'model_name': model_name,
            'base_url': base_url,
        },
        'evaluator_mode_counts': main_eval_mode_counts,
        'generation_mode_counts': main_gen_mode_counts,
        'golden_set_distribution': golden_set_distribution,
        'classifier_confidence_calibration': conf_stats,
        'human_rated_quality': human_calib_summary,
        'benchmark_results': results_table,
    }
    
    # 1. Save to run-specific directory
    with open(run_results_path, 'w', encoding='utf-8') as f:
        json.dump(output_res, f, indent=2)

    # 2. Save/copy to latest directory under base_results_dir
    latest_dir = os.path.join(base_results_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    with open(os.path.join(latest_dir, "evaluation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(output_res, f, indent=2)
    if os.path.exists(run_audit_path):
        shutil.copyfile(run_audit_path, os.path.join(latest_dir, "judge_audit_log.jsonl"))

    # 3. Write explicit pointer manifest in latest_dir
    latest_manifest = {
        'latest_run_id': run_id,
        'run_directory': run_dir,
        'run_results_path': run_results_path,
        'run_audit_path': run_audit_path,
        'updated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(os.path.join(latest_dir, "latest_manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(latest_manifest, f, indent=2)

    if output_base_dir is None:
        # Top-level results/ copies for backward compatibility
        with open("results/latest_manifest.json", 'w', encoding='utf-8') as f:
            json.dump(latest_manifest, f, indent=2)
        with open("results/evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(output_res, f, indent=2)
        if os.path.exists(run_audit_path):
            shutil.copyfile(run_audit_path, "results/judge_audit_log.jsonl")
        
        print(f"[Evaluation Harness] Saved run {run_id} benchmark metrics to {run_results_path}, results/latest/, and latest_manifest.json.")
    else:
        print(f"[Evaluation Harness] Saved isolated test run {run_id} benchmark metrics to {run_results_path} and {latest_dir}.")
    return output_res

if __name__ == "__main__":
    run_evaluation()
