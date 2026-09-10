import os
import json
import yaml
import numpy as np
from typing import List, Dict, Any

INTENT_TAXONOMY = [
    'order_status',
    'delivery_delay',
    'refund_request',
    'damaged_wrong_item',
    'account_access',
    'cancellation',
    'complaint_about_service',
    'general_inquiry'
]

def export_candidates_for_annotation(processed_path: str = "data/processed/amazonhelp_threads.jsonl",
                                     output_path: str = "golden/candidate_export.jsonl",
                                     target_size: int = 200) -> List[Dict[str, Any]]:
    """
    Exports unannotated candidate threads sampled from dataset for manual human annotation.
    """
    print(f"[Golden Set Ingestion] Sampling candidate threads for manual annotation from {processed_path}...")
    
    with open(processed_path, 'r', encoding='utf-8') as f:
        threads = [json.loads(line) for line in f]

    np.random.seed(42)
    indices = np.random.choice(len(threads), min(target_size, len(threads)), replace=False)
    candidates = [threads[i] for i in indices]

    export_items = []
    for idx, thread in enumerate(candidates, 1):
        export_items.append({
            'export_id': f"cand_{idx:03d}",
            'conversation_id': thread['conversation_id'],
            'customer_message': thread['customer_message'],
            'context_messages': thread['context_messages'],
            'original_brand_reply': thread['brand_reply'],
            'status': "UNANNOTATED_CANDIDATE"
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in export_items:
            f.write(json.dumps(item) + '\n')

    print(f"[Golden Set Export] Exported {len(export_items)} unannotated candidates to {output_path}.")
    return export_items

def load_and_verify_golden_set(golden_path: str = "golden/golden_set.jsonl") -> List[Dict[str, Any]]:
    """
    Loads and verifies the held-out reference evaluation dataset (N=200).
    Validates item schema, presence of observed historical human brand replies,
    and reference category assignments without running runtime programmatic rule generators.
    """
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Evaluation set file not found at {golden_path}.")

    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_items = [json.loads(line) for line in f]

    print(f"[Reference Set Verification] Verifying {len(golden_items)} held-out evaluation items...")

    for item in golden_items:
        if 'gold_intent' not in item or 'gold_escalate' not in item or 'gold_reply' not in item:
            raise ValueError(f"Evaluation set item {item.get('id')} has an incomplete label schema.")
        if not item.get('original_brand_reply'):
            raise ValueError(f"Evaluation set item {item.get('id')} missing original_brand_reply.")

    print(f"[Reference Set Verification] Verification PASSED. Loaded {len(golden_items)} held-out reference evaluation items with valid schema & historical brand replies.")
    return golden_items

if __name__ == "__main__":
    load_and_verify_golden_set()
