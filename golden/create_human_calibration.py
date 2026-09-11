import os
import json
import numpy as np
from typing import List, Dict, Any

def sample_calibration_candidates(golden_path: str = "golden/golden_set.jsonl",
                                 export_path: str = "golden/calibration_export.jsonl",
                                 sample_size: int = 50) -> List[Dict[str, Any]]:
    """
    Samples candidate items and exports unpopulated rubric forms for external human raters.
    Does NOT generate scores.
    """
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden evaluation set missing at {golden_path}.")

    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_items = [json.loads(line) for line in f]

    np.random.seed(123)
    indices = np.random.choice(len(golden_items), min(sample_size, len(golden_items)), replace=False)
    sampled = [golden_items[i] for i in indices]

    export_records = []
    for idx, item in enumerate(sampled, start=1):
        export_records.append({
            'item_id': f'calib_{idx:03d}',
            'golden_ref_id': item.get('id'),
            'conversation_id': item.get('conversation_id'),
            'customer_message': item.get('customer_message'),
            'context_messages': item.get('context_messages', []),
            'gold_reply': item.get('gold_reply'),
            'human_rubric_scores': {
                'correctness': None,
                'tone': None,
                'actionability': None,
                'safety': None
            },
            'human_total_score': None,
            'annotator_id': None,
            'adjudication_note': ""
        })

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        for rec in export_records:
            f.write(json.dumps(rec) + '\n')

    print(f"[Calibration Sampler] Exported {len(export_records)} unpopulated calibration forms to {export_path}.")
    return export_records

def verify_calibration_dataset(calibration_path: str = "golden/human_calibration.json") -> List[Dict[str, Any]]:
    """
    Validates an existing external calibration dataset (N=50 records).
    Does NOT generate scores or invent annotations.
    """
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found at {calibration_path}.")

    records = []
    with open(calibration_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"[Calibration Validation] Validating {len(records)} calibration annotation records...")
    
    for idx, item in enumerate(records, 1):
        if not item.get('annotator_id'):
            raise ValueError(f"Calibration item {item.get('item_id')} missing annotator_id.")
        if 'human_total_score' not in item or 'human_rubric_scores' not in item:
            raise ValueError(f"Calibration item {item.get('item_id')} missing human rubric scores.")
        
        # Verify rubric sum matches total
        rubric = item['human_rubric_scores']
        calc_total = float(rubric.get('correctness', 0) + rubric.get('tone', 0) + rubric.get('actionability', 0) + rubric.get('safety', 0))
        if calc_total != item['human_total_score']:
            raise ValueError(f"Calibration item {item.get('item_id')} score mismatch: rubric sum {calc_total} != total {item['human_total_score']}.")

    print(f"[Calibration Validation] Validation PASSED. Verified {len(records)} calibration annotation records.")
    return records

if __name__ == "__main__":
    verify_calibration_dataset()

