import re
import yaml
from typing import List, Dict, Any, Tuple

class EscalationEngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        esc_cfg = config.get('escalation', {})
        self.weights = esc_cfg.get('weights', {
            'intent_risk': 0.30,
            'low_confidence': 0.25,
            'keyword_risk': 0.20,
            'low_retrieval_sim': 0.10,
            'thread_length': 0.08,
            'sentiment_risk': 0.07
        })
        self.score_threshold = esc_cfg.get('score_threshold', 0.50)
        self.high_risk_keywords = esc_cfg.get('high_risk_keywords', [
            'lawyer', 'sue', 'legal', 'fraud', 'stolen', 'police', 'injury', 'scam', 'court', 'attorney'
        ])
        
        default_intent_risk_map = {
            'complaint_about_service': 0.70,
            'refund_request': 0.50,
            'damaged_wrong_item': 0.50,
            'account_access': 0.40,
            'cancellation': 0.30,
            'delivery_delay': 0.30,
            'order_status': 0.10,
            'general_inquiry': 0.10
        }
        self.intent_risk_map = esc_cfg.get('intent_risk_map', default_intent_risk_map)

    def evaluate(self, customer_message: str, intent: str, intent_confidence: float,
                 context_messages: List[str] = None, retrieval_score: float = 0.0) -> Tuple[bool, float, str]:
        """
        Evaluates multi-signal risk score including retrieval similarity and returns complete explainable reasons.
        Enforces Unified Escalation Policy across guidelines, annotation, and live engine:
        - Critical risk keywords or human agent requests -> immediate escalation.
        - Thread context exhaustion: 2+ prior turns AND unresolved dissatisfaction keywords -> escalation.
        - Severe complaints / high risk scores -> escalation via multi-signal threshold.
        """
        if context_messages is None:
            context_messages = []
            
        text_lower = customer_message.lower()
        
        # 1. Hard Keyword Risk Check (Word Boundaries Enforced)
        for kw in self.high_risk_keywords:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            if re.search(pattern, text_lower):
                return True, 1.0, f"Critical high-risk keyword detected: '{kw}'"
                
        # Explicit Human Request Trigger Check (Word Boundaries & Phrase Context Enforced)
        human_request_phrases = [
            r"\b(speak|talk|connect|transfer|get|need)\s+(me\s+)?(to|with)?\s*(a|an|the|any)?\s*(agent|human|representative|manager|supervisor|person)\b",
            r"\bhuman representative\b",
            r"\blive representative\b",
            r"\bcustomer service representative\b",
            r"\breal person\b",
            r"\bhuman agent\b",
            r"\bhuman support\b",
            r"\b(speak|talk)\s+to\s+(a|an)?\s*(manager|supervisor)\b",
            r"\b(want|need|get)\s+(a|an)?\s*(human|representative|agent|manager)\b",
            r"\btransfer me\b",
            r"\bconnect me\b"
        ]

        for pat in human_request_phrases:
            match = re.search(pat, text_lower)
            if match:
                matched_phrase = match.group(0).strip()
                return True, 0.95, f"Customer explicitly requested a human agent (triggered by '{matched_phrase}')"

        # 2. Multi-Signal Scoring Computation
        intent_risk = self.intent_risk_map.get(intent, 0.20)
        low_conf_risk = max(0.0, 1.0 - intent_confidence)
        keyword_risk = 0.80 if any(w in text_lower for w in ['unacceptable', 'disgusted', 'threw', 'stolen', 'worst']) else 0.0
        
        # Low historical retrieval similarity signal (if similarity < 0.35 => high risk)
        low_retrieval_risk = max(0.0, 1.0 - (retrieval_score / 0.35)) if retrieval_score < 0.35 else 0.0
        
        thread_length_risk = min(1.0, len(context_messages) / 3.0)
        sentiment_risk = 0.70 if any(w in text_lower for w in ['angry', 'furious', 'upset', 'terrible', 'horrible']) else 0.10

        risk_score = (
            self.weights.get('intent_risk', 0.30) * intent_risk +
            self.weights.get('low_confidence', 0.25) * low_conf_risk +
            self.weights.get('keyword_risk', 0.20) * keyword_risk +
            self.weights.get('low_retrieval_sim', 0.10) * low_retrieval_risk +
            self.weights.get('thread_length', 0.08) * thread_length_risk +
            self.weights.get('sentiment_risk', 0.07) * sentiment_risk
        )

        # 3. Unified Thread Context Exhaustion Policy: 2+ prior turns AND repeated unresolved dissatisfaction
        repetition_terms = [
            'again', 'still', 'second time', 'third time', 'already', 'repeatedly', 'multiple times'
        ]
        negative_status_terms = [
            'not fixed', 'no response', 'unresolved', "didn't help", 'did not help', 'no update',
            "haven't received", 'have not received', 'still waiting', 'no progress', 'not working', 'failed', 'stuck'
        ]
        
        customer_context = [m for m in context_messages if not m.lower().startswith(("brand:", "agent:", "support:"))]
        text_for_exhaustion = " ".join(customer_context + [customer_message]).lower()
        
        has_repetition = any(kw in text_for_exhaustion for kw in repetition_terms)
        has_negative_status = any(kw in text_for_exhaustion for kw in negative_status_terms)
        has_unresolved_dissatisfaction = has_repetition and has_negative_status

        if len(context_messages) >= 2 and has_unresolved_dissatisfaction:
            return True, max(risk_score, 0.75), "Thread context exhaustion: 2+ prior turns with repeated unresolved dissatisfaction"

        if risk_score >= self.score_threshold:
            reasons = []
            if intent_risk > 0.5:
                reasons.append(f"high-risk intent '{intent}'")
            if low_conf_risk > 0.4:
                reasons.append("low classifier confidence")
            if keyword_risk > 0:
                reasons.append("negative sentiment terms")
            if low_retrieval_risk > 0.5:
                reasons.append("low historical support interaction match")
            if thread_length_risk > 0.5:
                reasons.append("extended thread depth")
            if sentiment_risk > 0.5:
                reasons.append("high customer frustration/sentiment risk")
                
            reason_str = f"Escalated due to: {', '.join(reasons) if reasons else 'multi-signal risk threshold exceeded'}"
            return True, float(risk_score), reason_str

        return False, float(risk_score), None
