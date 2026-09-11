import os
import re
import json
import yaml
import pandas as pd
from typing import List, Dict, Any

def normalize_text(text: str, brand_handle: str = "@AmazonHelp") -> str:
    """Normalizes text by replacing URLs with <URL> and mentions (except brand) with <USER>."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    words = text.split()
    normalized_words = []
    brand_lower = brand_handle.lower()
    
    for word in words:
        if word.startswith('@'):
            clean_word = re.sub(r'[^\w@]', '', word)
            if clean_word.lower() == brand_lower:
                normalized_words.append(brand_handle)
            else:
                normalized_words.append('<USER>')
        else:
            normalized_words.append(word)
            
    return " ".join(normalized_words)

def to_bool(value) -> bool:
    """Safely coerces schema boolean values (bool, 'True', 'False', 1, 0) to bool."""
    if isinstance(value, bool):
        return value
    val_str = str(value).strip().lower()
    if val_str in {"true", "1"}:
        return True
    elif val_str in {"false", "0"}:
        return False
    raise ValueError(f"[Data Cleaning Error] Invalid boolean value for inbound: {value!r}")

def build_threads_for_brand(df: pd.DataFrame, brand_handle: str = "@AmazonHelp", max_threads: int = 25000, max_history_turns: int = 3) -> List[Dict[str, Any]]:
    """
    Extracts multi-turn conversation threads where customer mentions brand and brand replies.
    
    Traverses complete raw conversation graph (indexing all raw tweets before brand filtering)
    so ancestor customer turns that do not explicitly contain brand mentions are recovered.
    `conversation_id` / `interaction_id` represents the unique turn-level interaction pair (`conv_{inbound_tweet_id}`).
    """
    print(f"[Data Cleaning] Filtering dataset for brand {brand_handle}...")
    reply_col = 'in_response_to_tweet_id' if 'in_response_to_tweet_id' in df.columns else 'in_reply_to_tweet_id'
    brand_name = brand_handle.replace("@", "")
    
    # 1. Index ALL raw tweets to preserve full conversation graph ancestors
    tweet_dict = {}
    for idx, row in df.iterrows():
        t_id = str(row['tweet_id'])
        in_resp = str(row[reply_col]) if pd.notna(row[reply_col]) else None
        resp_tweet = str(row['response_tweet_id']) if pd.notna(row['response_tweet_id']) else None
        
        tweet_dict[t_id] = {
            'tweet_id': t_id,
            'author_id': str(row['author_id']),
            'inbound': to_bool(row['inbound']),
            'created_at': str(row.get('created_at', '')),
            'text': str(row['text']),
            'in_response_to_tweet_id': in_resp,
            'response_tweet_id': resp_tweet
        }
        
    print(f"[Data Cleaning] Indexed {len(tweet_dict)} total raw tweets for graph traversal.")
    
    threads = []
    seen_conversations = set()
    
    for tweet_id, tweet in tweet_dict.items():
        # Identify brand interaction endpoints: customer inbound mentioning brand with a brand reply
        mentions_brand = brand_handle.lower() in tweet['text'].lower()
        if tweet['inbound'] and mentions_brand and tweet['response_tweet_id']:
            response_ids = [r.strip() for r in str(tweet['response_tweet_id']).split(',') if r.strip()]
            
            selected_resp_id = None
            brand_reply_text = None
            
            # Selection rule: First matching chronological brand reply
            for resp_id in response_ids:
                if resp_id in tweet_dict and not tweet_dict[resp_id]['inbound']:
                    resp_author = tweet_dict[resp_id]['author_id'].lower()
                    if resp_author == brand_name.lower():
                        selected_resp_id = resp_id
                        brand_reply_text = tweet_dict[resp_id]['text']
                        break
                    
            if not brand_reply_text:
                continue
                
            # Traverse ancestor turns in FULL graph (depth <= max_history_turns)
            context_turns = []
            curr_reply_to = tweet['in_response_to_tweet_id']
            depth = 0
            
            while curr_reply_to and curr_reply_to in tweet_dict and depth < max_history_turns:
                prior_tweet = tweet_dict[curr_reply_to]
                role = "Brand" if not prior_tweet['inbound'] else "Customer"
                norm_prior = normalize_text(prior_tweet['text'], brand_handle)
                context_turns.insert(0, f"{role}: {norm_prior}")
                curr_reply_to = prior_tweet['in_response_to_tweet_id']
                depth += 1
                
            customer_text_norm = normalize_text(tweet['text'], brand_handle)
            brand_reply_norm = normalize_text(brand_reply_text, brand_handle)
            
            if len(customer_text_norm.split()) < 3 or len(brand_reply_norm.split()) < 3:
                continue
                
            conv_id = f"conv_{tweet['tweet_id']}"
            if conv_id in seen_conversations:
                continue
            seen_conversations.add(conv_id)
            
            threads.append({
                'id': conv_id,
                'conversation_id': conv_id,
                'interaction_id': conv_id,
                'turn_inbound_tweet_id': tweet['tweet_id'],
                'selected_reply_tweet_id': selected_resp_id,
                'response_selection_rule': 'first_chronological_brand_reply',
                'customer_message': customer_text_norm,
                'context_messages': context_turns,
                'brand_reply': brand_reply_norm,
                'raw_customer_text': tweet['text'],
                'raw_brand_reply': brand_reply_text
            })
            
            if len(threads) >= max_threads:
                break
                
    print(f"[Data Cleaning] Successfully extracted {len(threads)} clean multi-turn threads for {brand_handle}.")
    return threads

def run_pipeline(config_path="configs/config.yaml", brand_handle: str = None) -> str:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_path = config['paths']['raw_data']
    processed_path = config['paths']['processed_data']
    if brand_handle is None:
        brand_handle = config['brand']['handle']
    max_threads = config['data']['max_dev_threads']
    max_history_turns = config['data'].get('max_thread_history_turns', 3)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file {raw_path} not found. Please run src/download_data.py first.")
        
    print(f"[Data Cleaning] Loading raw CSV from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    
    threads = build_threads_for_brand(df, brand_handle=brand_handle, max_threads=max_threads, max_history_turns=max_history_turns)
    
    resolved_handle = brand_handle
    if len(threads) < 100:
        raise ValueError(
            f"[Data Cleaning Error] Insufficient thread data for brand '{brand_handle}': produced only {len(threads)} clean threads (minimum 100 required). "
            f"Silent fallback brand switching has been disabled to prevent domain task contamination. "
            f"Please verify raw dataset contains sufficient tweets for '{brand_handle}'."
        )
        
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, 'w', encoding='utf-8') as f:
        for thread in threads:
            f.write(json.dumps(thread) + '\n')
            
    # Save Resolved Brand Metadata to prevent cross-brand contamination downstream
    brand_meta_path = os.path.join(os.path.dirname(processed_path), "brand_metadata.json")
    with open(brand_meta_path, 'w', encoding='utf-8') as f:
        json.dump({'resolved_brand_handle': resolved_handle, 'thread_count': len(threads)}, f, indent=2)
        
    print(f"[Data Cleaning] Processed dataset saved to {processed_path} ({len(threads)} items, brand={resolved_handle}). Metadata: {brand_meta_path}")
    return processed_path

if __name__ == "__main__":
    run_pipeline()
