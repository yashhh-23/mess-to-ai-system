import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalSupportInteractionRetriever:
    """
    TF-IDF Lexical Retriever for observed historical support interaction pairs.
    Indexes historical customer messages and context threads (excluding answer text),
    attaching historical brand replies as document payload for evidence synthesis.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.items: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index(self, processed_path: str = "data/processed/amazonhelp_threads.jsonl",
                    golden_path: str = "golden/golden_set.jsonl"):
        print("[Retriever] Building Historical TF-IDF Lexical Index with Data Leakage Guard...")
        
        # Golden Set Exclusion Filter & Near-Duplicate Split Constraint
        golden_ids = set()
        golden_msgs = []
        if os.path.exists(golden_path):
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden_items = [json.loads(line) for line in f]
                golden_ids = {g['conversation_id'] for g in golden_items}
                golden_msgs = [g['customer_message'] for g in golden_items]
        print(f"[Retriever] Excluding {len(golden_ids)} Golden Set conversation IDs from RAG memory index.")

        with open(processed_path, 'r', encoding='utf-8') as f:
            threads = [json.loads(line) for line in f]

        excluded_near_dupe_count = 0
        if golden_msgs:
            dupe_vec = TfidfVectorizer(ngram_range=(1, 2)).fit(golden_msgs)
            X_gold = dupe_vec.transform(golden_msgs)

        self.items = []
        corpus_texts = []

        for thread in threads:
            if thread['conversation_id'] in golden_ids:
                continue
            
            if golden_msgs:
                X_cand = dupe_vec.transform([thread['customer_message']])
                max_sim = float(X_cand.dot(X_gold.T).toarray().max())
                if max_sim >= 0.85:
                    excluded_near_dupe_count += 1
                    continue

            # Customer issue + thread context as document representation (brand reply attached as payload)
            context_str = " ".join(thread.get('context_messages', []))
            text_to_index = f"{context_str} {thread['customer_message']}".strip()
            corpus_texts.append(text_to_index)
            self.items.append(thread)

        if excluded_near_dupe_count > 0:
            print(f"[Data Leakage Guard] Enforced split constraint: excluded {excluded_near_dupe_count} near-duplicate threads (similarity >= 0.85) from RAG memory index.")

        print(f"[Retriever] Indexing {len(self.items)} observed historical support interaction pairs...")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        self.is_indexed = True
        print("[Retriever] TF-IDF lexical index construction complete.")

    def retrieve(self, customer_message: str, context_messages: List[str] = None, top_k: int = 3, min_score: float = 0.0001) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K most lexically similar observed historical support interaction pairs using TF-IDF and cosine similarity.
        Filters out items with similarity score <= min_score to prevent zero-similarity arbitrary items from leaking into prompts.
        """
        if not self.is_indexed:
            return []

        context_str = " ".join(context_messages or [])
        query_text = f"{context_str} {customer_message}".strip()

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= min_score:
                continue
            item = self.items[idx]
            results.append({
                'score': score,
                'customer_message': item['customer_message'],
                'brand_reply': item['brand_reply'],
                'conversation_id': item['conversation_id']
            })
            
        return results

    def save(self, model_path: str = "models/vector_store.pkl"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'items': self.items, 'tfidf_matrix': self.tfidf_matrix, 'is_indexed': self.is_indexed}, f)
        print(f"[Retriever] Saved index to {model_path}.")

    def load(self, model_path: str = "models/vector_store.pkl"):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.items = data['items']
                self.tfidf_matrix = data['tfidf_matrix']
                self.is_indexed = data['is_indexed']
            print(f"[Retriever] Loaded index from {model_path}.")
            return True
        return False


# Backwards compatibility alias
HistoricalResolutionRetriever = HistoricalSupportInteractionRetriever


if __name__ == "__main__":
    retriever = HistoricalSupportInteractionRetriever()
    retriever.build_index()
    retriever.save()
    results = retriever.retrieve("My package was stolen or not delivered yesterday")
    print("Top Retrieval:", results[0] if results else "None")
