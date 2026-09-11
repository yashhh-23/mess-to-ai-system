# Reference Evaluation Set Annotation Guidelines for @AmazonHelp Twitter Support

## Overview
This document specifies the exact rules and guidelines used to construct the **200-Item Held-Out Reference Evaluation Set** (`golden/golden_set.jsonl`). Reference targets are derived from rule-assisted taxonomy mapping (`rule_assisted_reference_labeling`) and observed historical Twitter brand replies (`original_brand_reply`).

Each item in the golden set contains:
1. `id`: Unique conversation identifier.
2. `conversation_id`: Raw thread identifier (used for strict data leakage partitioning).
3. `customer_message`: Normalized current customer message.
4. `context_messages`: Chronological array of prior conversation turns formatted as `Customer: ...` or `Brand: ...`.
5. `gold_intent`: Hand-validated ground truth intent category.
6. `gold_escalate`: Ground truth binary escalation decision (`true` or `false`).
7. `gold_reply`: Reference high-quality brand reply written in `@AmazonHelp` voice.
8. `escalation_reason`: Human annotation explaining why escalation was chosen (or `null` if auto-handled).

---

## 1. Intent Labeling Guidelines (`gold_intent`)

Select exactly ONE primary intent from the 8 domain-designed and EDA-validated categories:

| Intent Label | Definition & Typical Phrases | Borderline / Edge Cases |
|---|---|---|
| `order_status` | Inquiries asking where package is, estimated delivery date, or tracking link requests. *"Where is my package?"*, *"Can I get tracking info?"* | If package is overdue past estimated date, mark as `delivery_delay`. |
| `delivery_delay` | Complaints about packages not arriving on expected date or delayed shipment. *"My order was supposed to arrive yesterday"*, *"Delayed in transit"*. | If customer explicitly demands money back due to delay, mark as `refund_request`. |
| `refund_request` | Demands for refund, money back, credit, or queries on refund status. *"I want a refund"*, *"Where is my money?"* | If asking how to return a broken item, mark as `damaged_wrong_item`. |
| `damaged_wrong_item` | Reports of broken goods, wrong items received, missing parts/box. *"Item arrived broken"*, *"Wrong size delivered"*. | If complaint is about driver dropping box, mark as `complaint_about_service`. |
| `account_access` | Password resets, prime subscription billing/cancellation, account locked. *"Can't log in"*, *"Charged for Prime unexpectedly"*. | If asking to cancel an order, mark as `cancellation`. |
| `cancellation` | Requests to cancel an active order or item before/during delivery. *"Please cancel order #123"*, *"Cancel my shipment"*. | If requesting refund for delivered item, mark as `refund_request`. |
| `complaint_about_service` | Driver misconduct, rude phone agent, lost package negligence, or brand service complaints. *"Your driver threw package"*, *"Terrible service"*. | Routine complaints are evaluated via multi-signal risk scoring; severe complaints with risk keywords ("lawyer", "sue", "stolen", "fraud") escalate (`gold_escalate=true`). |
| `general_inquiry` | Store policies, gift card queries, stock availability, seller questions. *"Do you accept Apple Pay?"*, *"Is item in stock?"* | General vague greetings without issue detail. |

---

## 2. Escalation Guidelines (`gold_escalate`)

Assign `gold_escalate = true` if ANY of the following criteria are met:

1. **Safety / Risk Keywords**: Customer text mentions legal action (*"lawyer"*, *"sue"*, *"court"*), fraud/scam (*"fraud"*, *"stolen card"*, *"hacked"*), physical injury (*"injured"*, *"hurt"*), or police involvement.
2. **Severe Emotional Distress / Agent Misconduct**: Extreme anger, severe agent/driver misconduct, or explicit threats to leave platform.
3. **Explicit Agent Request**: Customer explicitly requests a human representative (*"human"*, *"representative"*, *"agent"*, *"real person"*).
4. **Unresolved Thread Context Exhaustion**: Thread context shows 2+ prior turns AND evidence of repeated dissatisfaction or unresolved issue (*"still waiting"*, *"not fixed"*, *"already shared"*, *"third time"*).
5. **Sensitive Personal Information**: Issue requires sharing account details, passwords, or personal credentials that cannot be handled via public Twitter self-serve links.

Assign `gold_escalate = false` if:
- The issue is a standard informational request (tracking link, standard return policy, standard cancellation link).
- The issue can be resolved cleanly by directing the customer to `@AmazonHelp` official self-serve help links or standard DM form (`<URL>`).

---

## 3. Reference Reply Guidelines (`gold_reply`)

Gold replies should strictly match `@AmazonHelp` historical Twitter standards:
- **Tone**: Empathetic, polite, concise, professional.
- **Formatting**: Include `<USER>` handle at start, replace URLs with `<URL>`.
- **Actionability**: Direct customer to secure DM or official Amazon help link for account lookup.
- **Safety Rule**: Never ask for passwords or credit card numbers in replies. Never guarantee exact refund amounts without agent verification.

> [!NOTE]
> Reference replies in `golden_set.jsonl` (`gold_reply`) are set to observed historical human Twitter support responses (`original_brand_reply`) tied directly to the customer issue from the TWCS dataset. This ensures response evaluation measures alignment against authentic historical support responses rather than synthetic project templates.

*Example Gold Reply (`escalate = false`)*:
> "Hi <USER>, we're sorry to hear about the delay with your order! Please send us a quick DM with your order ID via this link: <URL> so we can look into this for you right away."

*Example Gold Reply (`escalate = true`)*:
> "Hi <USER>, we take reports of driver misconduct very seriously. Please reach out to our specialist support team directly via DM here: <URL> so we can escalate this to management."
