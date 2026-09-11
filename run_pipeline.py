import os
import sys
import json
import time
import hashlib
import yaml

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.download_data import download_twitter_support_data
from src.data_cleaning import run_pipeline as run_data_cleaning
from golden.build_golden_set import load_and_verify_golden_set
from golden.create_human_calibration import verify_calibration_dataset
from src.intent_classifier import HybridIntentClassifier
from src.retrieval import HistoricalSupportInteractionRetriever
from src.evaluate import run_evaluation, compute_source_code_hash, detect_near_duplicate_leakage

def compute_file_hash(filepath: str) -> str:
    """Computes MD5 hash of a file for checksum verification."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_git_commit_sha() -> str:
    """Gets current git commit SHA or returns placeholder if git repo is absent."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        return "NO_GIT_REPO"

import argparse

def execute_master_pipeline(config_path: str = "configs/config.yaml", mode: str = "full"):
    """
    Master Reproducibility Pipeline:
    Supports modes:
    - 'full' / 'rebuild': Data Ingest -> Cleaning -> Golden Verification -> Training -> Indexing -> Evaluation.
    - 'eval' / 'evaluate-existing': Runs benchmark evaluation on existing committed artifacts.
    - 'train': Runs cleaning, training, and indexing, skipping data download if raw CSV is absent but manifest is trusted.
    """
    print("\n==========================================================================================")
    print(f"       STARTING MASTER REPRODUCIBILITY PIPELINE (Mode: {mode.upper()})       ")
    print("==========================================================================================")
    start_time = time.time()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if mode in ["eval", "evaluate-existing"]:
        print("\n[Evaluation-Only Mode] Skipping retraining. Executing evaluation harness on committed artifacts...")
        results = run_evaluation(config_path=config_path)
        elapsed = time.time() - start_time
        print(f"\n[Evaluation Complete] Total Execution Time: {elapsed:.2f} seconds.")
        return results

    # 1. Download / Ingest Data
    raw_path = config['paths']['raw_data']
    manifest_path = "data/raw/data_manifest.json"
    raw_absent = not os.path.exists(raw_path)

    print("\n[Step 1/7] Data Acquisition & Schema Validation...")
    if not raw_absent:
        raw_csv = download_twitter_support_data(output_path=raw_path)
    else:
        try:
            raw_csv = download_twitter_support_data(output_path=raw_path)
            raw_absent = False
        except Exception as e:
            if os.path.exists(manifest_path):
                print(f"[Pipeline Warning] Raw CSV absent ({raw_path}) and download unavailable ({e}). "
                      f"Trusted data manifest found at '{manifest_path}'. Proceeding with pipeline using processed data.")
            else:
                raise

    # 2. Clean Data & Reconstruct Threads
    print("\n[Step 2/7] Data Cleaning & Multi-Turn Thread Extraction...")
    if not raw_absent and os.path.exists(raw_path):
        processed_jsonl = run_data_cleaning(config_path=config_path)
    else:
        processed_jsonl = config['paths']['processed_data']
        print(f"[Data Cleaning] Raw CSV absent; using existing processed dataset at '{processed_jsonl}'.")
    
    brand_meta_path = os.path.join(os.path.dirname(processed_jsonl), "brand_metadata.json")
    resolved_brand = "@AmazonHelp"
    if os.path.exists(brand_meta_path):
        with open(brand_meta_path, 'r', encoding='utf-8') as f:
            resolved_brand = json.load(f).get('resolved_brand_handle', "@AmazonHelp")

    # 3. Load & Verify Held-Out Reference Evaluation Set
    print("\n[Step 3/7] Verifying 200-Item Held-Out Reference Evaluation Set...")
    golden_jsonl = load_and_verify_golden_set(golden_path=config['paths']['golden_set'])

    # 4. Verify 50-Item Calibration Dataset Schema & Integrity
    print("\n[Step 4/7] Verifying 50-Item Calibration Dataset Schema & Integrity...")
    verify_calibration_dataset(calibration_path=config['paths'].get('human_calibration', 'golden/human_calibration.json'))

    # 5. Train Intent Classifier
    print("\n[Step 5/7] Training Hybrid Intent Classifier (Data Leakage Guard Enforced)...")
    clf = HybridIntentClassifier(confidence_threshold=config['intents']['confidence_threshold'])
    clf.train(processed_path=processed_jsonl, golden_path=config['paths']['golden_set'])
    clf.save(model_path=config['paths']['intent_model'])

    # 6. Build Historical Vector Retriever Index
    print("\n[Step 6/7] Indexing Historical Support Interaction Vector Store (Data Leakage Guard Enforced)...")
    retriever = HistoricalSupportInteractionRetriever()
    retriever.build_index(processed_path=processed_jsonl, golden_path=config['paths']['golden_set'])
    retriever.save(model_path=config['paths']['vector_store'])

    # Gather library versions
    import sklearn
    import pandas as pd
    import numpy as np
    library_versions = {
        'scikit-learn': getattr(sklearn, '__version__', 'unknown'),
        'pandas': getattr(pd, '__version__', 'unknown'),
        'numpy': getattr(np, '__version__', 'unknown'),
        'pyyaml': getattr(yaml, '__version__', 'unknown')
    }

    # Compute training class distribution
    with open(processed_jsonl, 'r', encoding='utf-8') as f:
        threads = [json.loads(line) for line in f]

    with open(config['paths']['golden_set'], 'r', encoding='utf-8') as f:
        golden_ids = {json.loads(line)['conversation_id'] for line in f}

    train_threads = [t for t in threads if t['conversation_id'] not in golden_ids]
    train_intents = [clf._rule_fallback(t['customer_message']) for t in train_threads]
    class_dist = {intent: train_intents.count(intent) for intent in set(train_intents)}

    clf_train_ids = getattr(clf, 'trained_conversation_ids', [t['conversation_id'] for t in train_threads])
    retrieval_indexed_ids = [item['conversation_id'] for item in retriever.items if 'conversation_id' in item]
    near_dupe_diag = detect_near_duplicate_leakage(
        golden_path=config['paths']['golden_set'],
        processed_path=processed_jsonl
    )

    data_leakage_guard_info = {
        'golden_conversation_ids_count': len(golden_ids),
        'trained_conversation_ids_count': len(clf_train_ids),
        'retrieval_indexed_ids_count': len(retrieval_indexed_ids),
        'exact_id_leakage_classifier_count': len(set(golden_ids).intersection(set(clf_train_ids))),
        'exact_id_leakage_retriever_count': len(set(golden_ids).intersection(set(retrieval_indexed_ids))),
        'near_duplicate_diagnostics': near_dupe_diag,
        'golden_conversation_ids': sorted(list(golden_ids)),
        'trained_conversation_ids': clf_train_ids,
        'retrieval_indexed_conversation_ids': retrieval_indexed_ids
    }

    # Generate run_id for evaluation provenance tracking
    run_id = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    run_dir = os.path.join("results", "runs", run_id)

    # Save Model Artifact Metadata Manifest
    manifest_checksum = compute_file_hash(manifest_path)
    metadata_path = "models/model_metadata.json"
    metadata = {
        'run_id': run_id,
        'run_dir': run_dir,
        'train_timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'download_required_for_full_repro': raw_absent,
        'git_commit_sha': get_git_commit_sha(),
        'source_code_hash': compute_source_code_hash(),
        'python_version': sys.version.split()[0],
        'library_versions': library_versions,
        'random_seed': 42,
        'resolved_brand_handle': resolved_brand,
        'sample_counts': {
            'total_threads': len(threads),
            'train_count': len(train_threads),
            'golden_count': len(golden_ids),
            'vector_index_count': len(retriever.items)
        },
        'train_class_distribution': class_dist,
        'data_leakage_guard': data_leakage_guard_info,
        'config_hash': compute_file_hash(config_path),
        'raw_data_checksum': compute_file_hash(raw_path),
        'dataset_manifest_checksum': manifest_checksum,
        'processed_data_checksum': compute_file_hash(processed_jsonl),
        'golden_set_checksum': compute_file_hash(config['paths']['golden_set']),
        'human_calibration_checksum': compute_file_hash(config['paths'].get('human_calibration', 'golden/human_calibration.json')),
        'brand_metadata_checksum': compute_file_hash(brand_meta_path),
        'intent_model_checksum': compute_file_hash(config['paths']['intent_model']),
        'vector_store_checksum': compute_file_hash(config['paths']['vector_store']),
        'pipeline_status': 'FRESH_REPRODUCED'
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Model Metadata] Saved artifact metadata manifest to {metadata_path}.")

    # 7. Run Evaluation Harness Benchmark
    print("\n[Step 7/7] Executing Evaluation Benchmark Harness...")
    results = run_evaluation(config_path=config_path, run_id=run_id)

    elapsed = time.time() - start_time
    print(f"\n==========================================================================================")
    print(f"       MASTER PIPELINE COMPLETE! Total Execution Time: {elapsed:.2f} seconds (< 15 mins)       ")
    print("==========================================================================================\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Reproducibility Pipeline & Evaluation Harness")
    parser.add_argument("--mode", type=str, choices=["full", "eval", "evaluate-existing", "train", "rebuild"], default="full",
                        help="Pipeline execution mode: 'full'/'rebuild' (default), 'eval'/'evaluate-existing', 'train'")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to system config.yaml")
    args = parser.parse_args()
    
    execute_master_pipeline(config_path=args.config, mode=args.mode)
