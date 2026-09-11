import re
import yaml
from typing import List, Dict, Any, Optional, Tuple, Literal
from src.llm_utils import LLMClient

class PostGenerationValidator:
    """
    Post-Generation Validation & Safety Layer.
    Enforces length limits, placeholder presence, channel-aware sensitive data policy, 
    and sanitizes unconditional operational promises.
    """
    @staticmethod
    def sanitize_and_validate(text: str, max_chars: int = 280, channel: Literal["public", "dm", "secure_form"] = "public") -> Tuple[str, Dict[str, bool]]:
        validation = {
            'length_pass': True,
            'placeholders_pass': True,
            'safety_pass': True,
            'guarantee_sanitized': False
        }

        sanitized_text = text or ""

        # 1. Sanitize Unconditional Promises
        promise_patterns = [
            # Original three patterns
            (r'we will refund (you|your money) (automatically|immediately)',
             'you can request a refund review via <URL>'),
            (r'free return label guaranteed',
             'you can request a return label via <URL>'),
            (r'100% money back guaranteed',
             'you can submit a claim via <URL>'),

            # Definite refund / money-back claims
            (r'you will receive a refund\b',
             'you may be eligible for a refund — please check via <URL>'),
            (r'your refund (is|has been|was) (approved|processed|issued|confirmed)',
             'your refund request can be reviewed via <URL>'),
            (r'(a |your )?refund has been (issued|sent|processed)',
             'a refund request can be submitted via <URL>'),

            # Definite delivery / arrival promises
            (r'your (package|order|item) will (arrive|be delivered)\b',
             'you can check your delivery status via <URL>'),
            (r'will (arrive|be delivered) (by|on|tomorrow|today)\b',
             'you can track your delivery via <URL>'),

            # Unconditional fix / resolution guarantees
            (r"we('ll| will) (fix|resolve|sort) this (immediately|right away|now|today)\b",
             'please DM us your order details and we\'ll look into this via <URL>'),
            (r'this (will|shall) be (resolved|fixed|sorted)\b',
             'please DM us and we\'ll investigate via <URL>'),

            # Free / automatic item / label guarantees
            (r'(a |your )?free (replacement|label) (has been|was) (issued|sent|approved)',
             'you can request a replacement or return label via <URL>'),
            (r'we guarantee a (replacement|refund|return)',
             'you can request a <URL> to check eligibility'),
        ]
        for pattern, replacement in promise_patterns:
            if re.search(pattern, sanitized_text, re.IGNORECASE):
                sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
                validation['guarantee_sanitized'] = True

        # 2. Enforce Greeting & Clean Body (strips duplicate greetings)
        text_clean = sanitized_text
        while True:
            prev = text_clean
            text_clean = re.sub(r"^\s*(hi|hello|hey|dear)\s+(@\w+|<USER>)\s*,?\s*", "", text_clean, flags=re.IGNORECASE)
            text_clean = re.sub(r"^\s*(hi|hello|hey|dear)\s*,?\s*", "", text_clean, flags=re.IGNORECASE)
            text_clean = re.sub(r"^\s*<USER>\s*,?\s*", "", text_clean, flags=re.IGNORECASE)
            text_clean = text_clean.strip()
            if text_clean == prev:
                break

        # Handle empty or greeting-only output
        if not text_clean or text_clean.lower() in [".", ",", "!"]:
            sanitized_text = "Hi <USER>, we’re here to help. Please visit <URL> for support options."
        else:
            sanitized_text = f"Hi <USER>, {text_clean}"

        # 3. Clean CTA Insertion for missing <URL>
        if "<URL>" not in sanitized_text:
            validation['placeholders_pass'] = False
            punct = "" if sanitized_text.rstrip().endswith((".", "?", "!")) else "."
            sanitized_text = f"{sanitized_text.rstrip()}{punct} Please visit <URL> for the next step."

        # 4. Channel-Aware Clause-Level Sensitive Data Policy & Safety Check
        safe_advisory_pattern = r"(do not|don't|never|avoid|not)\s+(share|post|tweet|send|provide|give)\b|for your security|keep your \w+ safe"
        
        # Highly sensitive credentials prohibited across ALL channels (public, dm, secure_form)
        strict_prohibited_credentials = [
            'password', 'passcode', 'credit card', 'debit card', 'card number', 'ssn', 'social security',
            'cvv', 'cvc', 'bank details', 'bank account', 'routing number', 'security code', 'otp',
            'verification code', 'pin number', 'government id', 'passport'
        ]
        
        # Public-only PII solicitation patterns (prohibited on public Twitter, allowed in DM/secure_form)
        public_pii_solicit_patterns = [
            r"send (us )?your (email|phone|address|password|pin|card)",
            r"tweet (us )?your (email|phone|address|password|pin|card)",
            r"reply with your (email|phone|address|password|pin|card)",
            r"provide your (email|phone|address|password|pin|card)",
            r"share your (email|phone|address|contact details)"
        ]

        clauses = re.split(r'[\.\!\?;\n—–|-]+', sanitized_text)
        is_safe = True
        for clause in clauses:
            clause_clean = clause.strip()
            if not clause_clean:
                continue
            clause_lower = clause_clean.lower()
            has_prohibited = any(term in clause_lower for term in strict_prohibited_credentials)
            has_public_solicit = (channel == "public") and any(re.search(pat, clause_lower) for pat in public_pii_solicit_patterns)

            if has_prohibited or has_public_solicit:
                is_clause_negated = bool(re.search(safe_advisory_pattern, clause_lower))
                if not is_clause_negated:
                    is_safe = False
                    break

        if not is_safe:
            validation['safety_pass'] = False
            if channel == "public":
                sanitized_text = "Hi <USER>, please do not share personal or payment details here. Send us a DM so we can look into this securely: <URL>."
            else:
                sanitized_text = "Hi <USER>, for your security, we cannot accept passwords or payment credentials in messages. Please visit our secure help center: <URL>."

        # 5. Enforce Character Length (<= 280 chars)
        if len(sanitized_text) > max_chars:
            validation['length_pass'] = False
            if "<URL>" in sanitized_text:
                body = sanitized_text.replace("<URL>", "").strip()
                cutoff = max_chars - len(" <URL>") - 3
                sanitized_text = body[:cutoff].rstrip() + "... <URL>"
            else:
                sanitized_text = sanitized_text[:max_chars - 10].rstrip() + "... <URL>"

        return sanitized_text.strip(), validation

class RAGReplyGenerator:
    """
    Retrieval-Augmented Generation (RAG) Reply Engine.
    Uses Top-K historical support interactions as evidence to synthesize responses via LLMClient,
    or via an evidence-adapted fallback engine when API keys are absent.
    Includes post-generation validation and citation tracing.
    """
    def __init__(self, brand_handle: str = "@AmazonHelp", provider: str = "auto", api_key: Optional[str] = None,
                 model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.brand_handle = brand_handle
        self.llm_client = LLMClient(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
        self.validator = PostGenerationValidator()

    def generate_reply_detailed(self, customer_message: str, intent: str, retrieved_pairs: List[Dict[str, Any]],
                                context_messages: List[str] = None, channel: Literal["public", "dm", "secure_form"] = "public") -> Dict[str, Any]:
        """
        Drafts a reply grounded in historical evidence and returns full metadata & citations.
        """
        if context_messages is None:
            context_messages = []

        context_str = "\n".join(context_messages) if context_messages else "None"
        retrieved_context_str = ""
        citations = []
        
        for i, pair in enumerate(retrieved_pairs[:3], 1):
            cid = pair.get('conversation_id', f'hist_{i}')
            score = round(float(pair.get('score', 0.0)), 4)
            citations.append({'conversation_id': cid, 'similarity_score': score})
            retrieved_context_str += f"\nEvidence {i} [ID: {cid}, Sim: {score}]:\nCustomer: {pair.get('customer_message','')}\nHistorical Resolution: {pair.get('brand_reply','')}\n"

        system_prompt = (
            f"You are the official Twitter customer support agent for {self.brand_handle}.\n"
            "Style rules:\n"
            "1. Tone: Empathetic, professional, direct, and concise (STRICTLY UNDER 280 CHARACTERS).\n"
            "2. Format: Always start with 'Hi <USER>,' and include URL placeholder '<URL>'.\n"
            "3. Safety: NEVER ask for passwords or make unverified operational guarantees.\n"
            "4. Grounding: Adapt evidence from retrieved examples without copying case-specific user names or numbers."
        )

        user_prompt = (
            f"Current Thread Context:\n{context_str}\n\n"
            f"New Customer Message: {customer_message}\n"
            f"Predicted Intent: {intent}\n\n"
            f"Retrieved Historical Evidence:\n{retrieved_context_str}\n\n"
            "Synthesize the brand response:"
        )

        llm_detail = self.llm_client.generate_detailed(user_prompt, system_prompt=system_prompt)
        
        raw_draft = None
        generation_mode = "Grounded-Template-Fallback"

        if llm_detail['content'] and not llm_detail['used_fallback']:
            raw_draft = llm_detail['content'].strip()
            generation_mode = f"LLM-RAG-Synthesized ({llm_detail['provider']} / {llm_detail['model_name']})"
        else:
            # Evidence-Driven Adaptation Fallback (No raw verbatim copy)
            best_match = retrieved_pairs[0] if retrieved_pairs else None
            best_score = best_match.get('score', 0) if best_match else 0.0

            if best_match and best_score >= 0.35:
                generation_mode = "Evidence-Adapted-Fallback"
                hist_text = best_match['brand_reply']
                # Scrub specific names/handles to treat as evidence template
                adapted = re.sub(r'@[A-Za-z0-9_]+', '<USER>', hist_text)
                adapted = re.sub(r'https?://\S+', '<URL>', adapted)
                raw_draft = adapted
            else:
                generation_mode = "Template-Fallback"
                templates = {
                    'order_status': "Hi <USER>, you can check the latest tracking updates and delivery status for your package here: <URL>. Let us know if you need any further help!",
                    'delivery_delay': "Hi <USER>, we're sorry for the delay! Please send us a quick DM with your order details via <URL> so we can check on its location for you.",
                    'refund_request': "Hi <USER>, we'd be glad to help check your refund status! Please send us a DM with your order ID here: <URL>.",
                    'damaged_wrong_item': "Hi <USER>, we're so sorry your item arrived damaged or incorrect! Please visit <URL> to start a return or replacement request for your order.",
                    'account_access': "Hi <USER>, for assistance with your account login or Prime subscription, please visit our secure help center here: <URL> or DM us.",
                    'cancellation': "Hi <USER>, if your order hasn't shipped yet, you can cancel it directly from your orders page: <URL>. Let us know if you need help!",
                    'complaint_about_service': "Hi <USER>, we apologize for this experience and want to make it right. Please send us a DM with your details via <URL>.",
                    'general_inquiry': "Hi <USER>, thanks for reaching out! You can view detailed information on our store policies and services here: <URL>."
                }
                raw_draft = templates.get(intent, templates['general_inquiry'])

        # Post-Generation Validation Layer
        validated_reply, val_flags = self.validator.sanitize_and_validate(raw_draft, max_chars=280, channel=channel)

        return {
            'reply': validated_reply,
            'raw_draft': raw_draft,
            'generation_mode': generation_mode,
            'used_fallback': llm_detail['used_fallback'],
            'citations': citations,
            'validation': val_flags
        }

    def generate_reply(self, customer_message: str, intent: str, retrieved_pairs: List[Dict[str, Any]],
                       context_messages: List[str] = None, channel: Literal["public", "dm", "secure_form"] = "public") -> str:
        """Convenience method returning post-validated reply string."""
        res = self.generate_reply_detailed(customer_message, intent, retrieved_pairs, context_messages, channel=channel)
        return res['reply']
