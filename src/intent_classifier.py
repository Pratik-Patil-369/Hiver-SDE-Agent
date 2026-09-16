"""Intent classifier.

Primary (offline, reproducible): TF-IDF + LogisticRegression trained on
data/processed/conversations.csv with weak labels from keyword rules, then
evaluated on the hand-labelled golden set. This keeps <15-min repro with no API key.

Optional (if GEMINI_API_KEY set): zero-shot LLM classifier used by the live agent.
The evaluation harness compares TF-IDF vs LLM so the report has real numbers.
"""
import json
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intents import INTENT_NAMES


def build_tfidf_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


KEYWORD_RULES = [
    ("baggage_issue", ["bag", "luggage", "suitcase", "baggage claim", "lost bag"]),
    ("refund_compensation", ["refund", "voucher", "compensation", "credit", "reimburse"]),
    ("booking_change", ["change my flight", "rebook", "cancel my", "reschedule", "standby", "itinerary"]),
    ("checkin_boarding", ["check in", "check-in", "boarding pass", "gate", "boarding group"]),
    ("seat_upgrade", ["seat", "upgrade", "legroom", "middle seat", "entertainment", "IFE"]),
    ("customer_service", ["rude", "unhelpful", "on hold", "hung up", "customer service", "agent was"]),
    ("loyalty_program", ["miles", "aadvantage", "elite", "status", "gold", "platinum", "points"]),
    ("website_app_issue", ["website", "app", "log in", "login", "crashes", "error", "payment failed"]),
    ("flight_delay_cancel", ["delayed", "delay", "cancelled", "canceled", "missed connection", "diverted", "stuck"]),
    ("flight_status_info", ["what time", "on time", "status of", "land?", "arrive", "depart"]),
]


def weak_label(text: str) -> str:
    t = text.lower()
    for intent, kws in KEYWORD_RULES:
        if any(k in t for k in kws):
            return intent
    return "flight_status_info"  # most generic fallback


def train_weakly_supervised(conversations: pd.DataFrame):
    """Train TF-IDF on weak labels so the model exists without golden labels leaking."""
    labels = conversations["customer_message"].map(weak_label)
    model = build_tfidf_model()
    model.fit(conversations["customer_message"], labels)
    return model


def predict_with_confidence(model, messages):
    proba = model.predict_proba(messages)
    classes = list(model.classes_)
    out = []
    for row in proba:
        i = int(row.argmax())
        out.append({"intent": classes[i], "confidence": float(row[i])})
    return out


def classify_llm(message: str, intents) -> dict:
    """Gemini zero-shot classifier. Raises if no API key."""
    import google.generativeai as genai
    from src.config import GEMINI_API_KEY, GEMINI_MODEL
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    intent_text = "\n".join(f"- {i['name']}: {i['definition']}" for i in intents)
    prompt = f"""You are a strict customer-support intent classifier. Classify, do not answer.
Available intents:
{intent_text}
Customer message: {message}
Rules: exactly one intent, valid JSON with intent/confidence(0-1)/reason."""
    resp = model.generate_content(prompt)
    text = resp.text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)
