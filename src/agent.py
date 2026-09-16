"""Complete agent: intent -> retrieval -> reply -> escalation."""
import pandas as pd
from src import config
from src.intents import INTENTS
from src.intent_classifier import train_weakly_supervised, predict_with_confidence, classify_llm
from src.embeddings import get_embedder
from src.retriever import Retriever
from src.response_generator import generate_reply
from src.escalation import decide_escalation


class Agent:
    def __init__(self, conversations: pd.DataFrame, use_llm_intent=False):
        self.conversations = conversations
        self.use_llm_intent = use_llm_intent and bool(config.GEMINI_API_KEY)
        self.clf = train_weakly_supervised(conversations)
        self.embedder, self.emb_kind = get_embedder("auto")
        self.embedder.fit(conversations["customer_message"].tolist())
        embs = self.embedder.encode(conversations["customer_message"].tolist())
        self.retriever = Retriever(embs, conversations)

    def predict_intent(self, message: str):
        if self.use_llm_intent:
            try:
                r = classify_llm(message, INTENTS)
                if r.get("intent") in [i["name"] for i in INTENTS]:
                    return r
            except Exception:
                pass
        p = predict_with_confidence(self.clf, [message])[0]
        return {"intent": p["intent"], "confidence": p["confidence"], "reason": "tfidf-lr"}

    def handle(self, message: str, k=5):
        intent_result = self.predict_intent(message)
        q = self.embedder.encode([message])[0]
        cases = self.retriever.search(q, k=k)
        reply = generate_reply(message, intent_result["intent"], cases)
        esc = decide_escalation(intent_result["confidence"], cases, message)
        return {"intent": intent_result, "historical_cases": cases,
                "reply": reply, "escalation": esc}
