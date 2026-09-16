"""Generate real data: data/processed/conversations.csv (500 real KB) + data/golden/golden_set.csv (200 real).

Uses authentic tweets from the Kaggle TWCS AmericanAir dataset.
Golden set is hand/Gemini labeled across the 10 data-derived intents + high-risk edge cases.
Zero overlap with the KB.
"""
import json
import os
import re
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import google.generativeai as genai
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intents import INTENTS

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-lite-latest")
raw_path = ROOT / "data" / "raw" / "americanair_twcs_sample.csv"
if not raw_path.exists():
    raise FileNotFoundError(f"Raw data missing at {raw_path}")

df = pd.read_csv(raw_path)
print(f"Loaded {len(df)} authentic AmericanAir conversations.")

# 1. Knowledge Base: First 500 conversations
kb = df.iloc[:500].copy()
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
kb.to_csv(ROOT / "data" / "processed" / "conversations.csv", index=False)
print("Saved 500 authentic AmericanAir conversations to data/processed/conversations.csv")

# 2. Golden Evaluation Set Candidates: From remaining pool (Disjoint from KB)
pool = df.iloc[500:].copy().reset_index(drop=True)
print(f"Candidate pool for evaluation set: {len(pool)} tweets.")

RISKY_TERMS = ["lawyer", "legal", "fraud", "scam", "police", "medical", "emergency",
               "safety", "threat", "discriminat", "lawsuit", "sue", "attorney"]

intents_text = "\n".join([f"- {i['name']}: {i['definition']}" for i in INTENTS])

prompt_template = """You are an expert customer-support annotator for American Airlines.
Classify this genuine customer tweet into exactly ONE of the 10 intents:
{intents_text}

Customer Tweet:
"{text}"

Decide:
1. intent: Must be one of the 10 exact intent names above.
2. escalation (true/false): Should this be escalated to a human agent? (true if medical emergency, safety crisis, legal threat/lawyer, police/fraud report, or multi-part complex rant; false if standard customer service or info request).
3. reason: A brief explanation of the intent classification.

Return JSON only:
{{
  "intent": "<name>",
  "escalation": false,
  "reason": "<explanation>"
}}
"""

target_per_intent = 18
target_escalations = 20
intent_counts = {i["name"]: 0 for i in INTENTS}
escalation_count = 0
labeled = []

print("Selecting and annotating 200 real golden examples across 10 intents...")

for idx, row in pool.iterrows():
    if len(labeled) >= 200:
        break
    text = str(row["customer_message"]).strip()
    words = text.split()
    if len(words) < 4 or len(words) > 85:
        continue

    low = text.lower()
    has_risk = any(r in low for r in RISKY_TERMS) or len(words) > 55

    try:
        p = prompt_template.format(intents_text=intents_text, text=text)
        resp = model.generate_content(p)
        raw_text = resp.text.strip()
        raw_text = re.sub(r"^```json\s*", "", raw_text)
        raw_text = re.sub(r"```$", "", raw_text)
        data = json.loads(raw_text)

        intent = data.get("intent")
        esc = bool(data.get("escalation")) or has_risk

        if intent not in intent_counts:
            continue

        take = False
        if esc and escalation_count < target_escalations:
            escalation_count += 1
            take = True
        elif not esc and intent_counts[intent] < target_per_intent:
            intent_counts[intent] += 1
            take = True

        if take:
            labeled.append({
                "id": len(labeled) + 1,
                "customer_message": text,
                "true_intent": intent,
                "true_escalation": esc
            })
            if len(labeled) % 20 == 0:
                print(f"Annotated {len(labeled)}/200 real examples...")
            time.sleep(0.1)
    except Exception as e:
        time.sleep(0.5)
        continue

# Fill any remaining slots if needed
if len(labeled) < 200:
    for idx, row in pool.iloc[len(labeled)+40:].iterrows():
        if len(labeled) >= 200:
            break
        text = str(row["customer_message"]).strip()
        if text in [x["customer_message"] for x in labeled]:
            continue
        try:
            p = prompt_template.format(intents_text=intents_text, text=text)
            resp = model.generate_content(p)
            raw_text = resp.text.strip()
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"```$", "", raw_text)
            data = json.loads(raw_text)
            intent = data.get("intent")
            if intent in intent_counts:
                labeled.append({
                    "id": len(labeled) + 1,
                    "customer_message": text,
                    "true_intent": intent,
                    "true_escalation": bool(data.get("escalation"))
                })
        except Exception:
            continue

df_golden = pd.DataFrame(labeled)
(ROOT / "data" / "golden").mkdir(parents=True, exist_ok=True)
df_golden.to_csv(ROOT / "data" / "golden" / "golden_set.csv", index=False)

print(f"\nCompleted! Saved data/golden/golden_set.csv with {len(df_golden)} real examples.")
print("Intent distribution on real data:")
print(df_golden["true_intent"].value_counts())
print(f"Escalation rate: {df_golden['true_escalation'].mean():.2f}")
