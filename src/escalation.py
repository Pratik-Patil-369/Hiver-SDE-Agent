"""Escalation: rule-based by default (transparent + testable), not pure LLM vibes."""
from src.config import INTENT_CONFIDENCE_THRESHOLD, RETRIEVAL_SIMILARITY_THRESHOLD

RISKY_TERMS = ["lawyer", "legal", "fraud", "scam", "chargeback", "police",
               "lawsuit", "sue", "attorney", "discriminat", "medical", "emergency",
               "safety", "threat", "harass"]


def decide_escalation(intent_confidence: float, retrieval_results, customer_message: str) -> dict:
    reasons = []
    if intent_confidence < INTENT_CONFIDENCE_THRESHOLD:
        reasons.append(f"Low intent confidence ({intent_confidence:.2f} < {INTENT_CONFIDENCE_THRESHOLD})")
    if not retrieval_results:
        reasons.append("No historical support evidence")
    elif retrieval_results[0]["score"] < RETRIEVAL_SIMILARITY_THRESHOLD:
        reasons.append(f"Weak historical similarity ({retrieval_results[0]['score']:.2f} < {RETRIEVAL_SIMILARITY_THRESHOLD})")
    low = customer_message.lower()
    hits = [t for t in RISKY_TERMS if t in low]
    if hits:
        reasons.append(f"Potential high-risk issue ({', '.join(hits)})")
    # very long / multi-issue messages are harder to auto-handle safely
    if len(customer_message.split()) > 60:
        reasons.append("Long multi-part message, needs human review")
    if reasons:
        return {"decision": "HUMAN", "reason": "; ".join(reasons)}
    return {"decision": "AUTO-HANDLE",
            "reason": "High-confidence intent with sufficient historical evidence"}
