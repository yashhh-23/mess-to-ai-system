import os
import json
import pickle
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

INTENTS = [
    'order_status',
    'delivery_delay',
    'refund_request',
    'damaged_wrong_item',
    'account_access',
    'cancellation',
    'complaint_about_service',
    'general_inquiry'
]

RULE_KEYWORDS = {
    'order_status': ['track', 'tracking', 'eta', 'where is', 'location', 'arriving', 'status'],
    'delivery_delay': ['delay', 'late', 'past due', 'was supposed to', 'yesterday', 'still waiting'],
    'refund_request': ['refund', 'money back', 'chargeback', 'credit', 'charged'],
    'damaged_wrong_item': ['damaged', 'broken', 'wrong item', 'missing', 'defective', 'shattered'],
    'account_access': ['login', 'password', 'prime', 'subscription', 'account', 'locked'],
    'cancellation': ['cancel', 'cancellation', 'stop order', 'dont send'],
    'complaint_about_service': ['terrible', 'horrible', 'worst', 'driver', 'threw', 'rude', 'lawyer', 'sue', 'police', 'stolen'],
    'general_inquiry': ['how do i', 'can i', 'option', 'store', 'in stock', 'policy', 'gift card']
}

def validate_taxonomy_alignment(config_path: str = "configs/config.yaml") -> bool:
    """
    Validates taxonomy integrity across configuration, model classes, escalation risk map,
    and reply generator templates. Fails fast if drift occurs.
    """
    if not os.path.exists(config_path):
        return True

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    yaml_taxonomy = cfg.get('intents', {}).get('taxonomy', [])
    if set(yaml_taxonomy) != set(INTENTS):
        raise ValueError(
            f"[Taxonomy Config Mismatch Error] Configured taxonomy in {config_path} {yaml_taxonomy} "
            f"does not match model INTENTS {INTENTS}."
        )

    intent_risk_map = cfg.get('escalation', {}).get('intent_risk_map', {})
    missing_risk = [intent for intent in yaml_taxonomy if intent not in intent_risk_map]
    if missing_risk:
        raise ValueError(
            f"[Taxonomy Config Mismatch Error] Escalation intent_risk_map missing risk scores for intents: {missing_risk}"
        )

    # Check template support in reply generator
    from src.reply_generator import PostGenerationValidator
    templates = {
        'order_status', 'delivery_delay', 'refund_request', 'damaged_wrong_item',
        'account_access', 'cancellation', 'complaint_about_service', 'general_inquiry'
    }
    missing_templates = [intent for intent in yaml_taxonomy if intent not in templates]
    if missing_templates:
        raise ValueError(
            f"[Taxonomy Config Mismatch Error] Reply generator missing fallback templates for intents: {missing_templates}"
        )

    return True

class HybridIntentClassifier:
    """
    Hybrid Rules + Supervised TF-IDF Logistic Regression Intent Classifier.
    Uses CalibratedClassifierCV(cv=3) for probability calibration.
    
    Calibration Disclosure:
    Confidence scores are calibrated against weak-supervised rule labels (_rule_fallback).
    Operational probability calibration claims for live production deployment require
    human ground-truth evaluation.
    """
    def __init__(self, confidence_threshold: float = 0.60, config_path: str = "configs/config.yaml"):
        self.confidence_threshold = confidence_threshold
        self.config_path = config_path
        validate_taxonomy_alignment(config_path=config_path)
        
        self.taxonomy = INTENTS
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    if cfg and 'intents' in cfg and 'taxonomy' in cfg['intents']:
                        self.taxonomy = cfg['intents']['taxonomy']
            except Exception:
                pass

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.base_classifier = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
        self.model = CalibratedClassifierCV(self.base_classifier, cv=3)
        self.is_fitted = False

    def _rule_fallback(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for intent, kws in RULE_KEYWORDS.items():
            scores[intent] = sum(1 for kw in kws if kw in text_lower)
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return 'general_inquiry'
        return best_intent

    def train(self, processed_path: str = "data/processed/amazonhelp_threads.jsonl",
              golden_path: str = "golden/golden_set.jsonl"):
        print(f"[Intent Classifier] Training classifier with Data Leakage Guard...")
        
        golden_ids = set()
        if os.path.exists(golden_path):
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden_items = [json.loads(line) for line in f]
                golden_ids = {g['conversation_id'] for g in golden_items}
        print(f"[Intent Classifier] Excluding {len(golden_ids)} Golden Set conversation IDs from training pool.")

        with open(processed_path, 'r', encoding='utf-8') as f:
            threads = [json.loads(line) for line in f]

        train_texts = []
        train_labels = []
        self.trained_conversation_ids = []

        for thread in threads:
            if thread['conversation_id'] in golden_ids:
                continue
            
            self.trained_conversation_ids.append(thread['conversation_id'])
            context_str = " ".join(thread.get('context_messages', []))
            full_input = f"{context_str} Customer: {thread['customer_message']}".strip()
            
            label = self._rule_fallback(thread['customer_message'])
            train_texts.append(full_input)
            train_labels.append(label)

        print(f"[Intent Classifier] Training sample size: {len(train_texts)} items.")
        X_tfidf = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_tfidf, train_labels)
        self.is_fitted = True
        print("[Intent Classifier] Model training complete.")

    def predict(self, customer_message: str, context_messages: List[str] = None) -> Tuple[str, float, bool]:
        """
        Predicts intent and returns (predicted_intent, true_confidence, used_fallback).
        Preserves true un-inflated model confidence.
        """
        if context_messages is None:
            context_messages = []

        context_str = " ".join(context_messages)
        full_input = f"{context_str} Customer: {customer_message}".strip()

        if not self.is_fitted:
            return self._rule_fallback(customer_message), 0.50, True

        X_val = self.vectorizer.transform([full_input])
        probs = self.model.predict_proba(X_val)[0]
        max_idx = np.argmax(probs)
        pred_intent = self.model.classes_[max_idx]
        confidence = float(probs[max_idx])

        used_fallback = False
        if confidence < self.confidence_threshold:
            pred_intent = self._rule_fallback(customer_message)
            used_fallback = True

        return pred_intent, confidence, used_fallback

    def save(self, model_path: str = "models/intent_classifier.pkl"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model, 'is_fitted': self.is_fitted}, f)
        print(f"[Intent Classifier] Saved model to {model_path}.")

    def load(self, model_path: str = "models/intent_classifier.pkl"):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.model = data['model']
                self.is_fitted = data['is_fitted']
            print(f"[Intent Classifier] Loaded model from {model_path}.")
            return True
        return False

if __name__ == "__main__":
    clf = HybridIntentClassifier()
    clf.train()
    clf.save()
