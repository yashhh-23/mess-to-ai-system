import json

def update_gold_replies_to_observed_human_replies(golden_path: str = "golden/golden_set.jsonl"):
    """
    Updates gold_reply in golden_set.jsonl to use authentic, case-specific observed 
    human brand replies (original_brand_reply) rather than static templates.
    """
    with open(golden_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]

    updated_items = []
    for item in items:
        orig = item.get('original_brand_reply', '')
        if orig:
            # Set gold_reply to authentic observed human brand reply
            item['gold_reply'] = orig
            item['reply_source'] = "observed_historical_human_reply"
        else:
            item['reply_source'] = "human_curated_reference"
        updated_items.append(item)

    with open(golden_path, 'w', encoding='utf-8') as f:
        for item in updated_items:
            f.write(json.dumps(item) + '\n')

    print(f"[Golden Set Update] Updated {len(updated_items)} items in {golden_path} to use case-specific observed human replies.")

if __name__ == "__main__":
    update_gold_replies_to_observed_human_replies()
