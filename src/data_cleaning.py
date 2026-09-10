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

def build_threads_for_brand(df: pd.DataFrame, brand_handle: str = "@AmazonHelp", max_threads: int = 25000, max_history_turns: int = 3) -> List[Dict[str, Any]]:
    """
    Extracts multi-turn conversation threads where customer mentions brand and brand replies.
    """
    print(f"[Data Cleaning] Filtering dataset for brand {brand_handle}...")
    reply_col = 'in_response_to_tweet_id' if 'in_response_to_tweet_id' in df.columns else 'in_reply_to_tweet_id'
    brand_name = brand_handle.replace("@", "")
    
    is_brand_author = df['author_id'].astype(str).str.lower() == brand_name.lower()
    mentions_brand = df['text'].astype(str).str.lower().str.contains(brand_handle.lower())
    
    brand_df = df[is_brand_author | mentions_brand].copy()
    print(f"[Data Cleaning] Total tweets referencing {brand_handle}: {len(brand_df)}")
    
    tweet_dict = {}
    for idx, row in brand_df.iterrows():
        t_id = str(row['tweet_id'])
        in_resp = str(row[reply_col]) if pd.notna(row[reply_col]) else None
        resp_tweet = str(row['response_tweet_id']) if pd.notna(row['response_tweet_id']) else None
        
        tweet_dict[t_id] = {
            'tweet_id': t_id,
            'author_id': str(row['author_id']),
            'inbound': bool(row['inbound']),
            'created_at': str(row.get('created_at', '')),
            'text': str(row['text']),
            'in_response_to_tweet_id': in_resp,
            'response_tweet_id': resp_tweet
        }
        
    threads = []
    seen_conversations = set()
    
    for tweet_id, tweet in tweet_dict.items():
        if tweet['inbound'] and tweet['response_tweet_id']:
            response_ids = [r.strip() for r in str(tweet['response_tweet_id']).split(',') if r.strip()]
            
            brand_reply_text = None
            for resp_id in response_ids:
                if resp_id in tweet_dict and not tweet_dict[resp_id]['inbound']:
                    brand_reply_text = tweet_dict[resp_id]['text']
                    break
                    
            if not brand_reply_text:
                continue
                
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
                'tweet_id': tweet['tweet_id'],
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

def run_pipeline(config_path="configs/config.yaml") -> str:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_path = config['paths']['raw_data']
    processed_path = config['paths']['processed_data']
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
            f"[Data Cleaning Error] Brand '{brand_handle}' produced only {len(threads)} clean threads (minimum 100 required). "
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
