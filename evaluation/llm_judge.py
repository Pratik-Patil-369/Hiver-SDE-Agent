"""LLM-as-judge with offline heuristic fallback.

Rubric (1-5): correctness, groundedness, helpfulness, tone, completeness.
- If GEMINI_API_KEY set: Gemini scores with strict grounding rules.
- Else: heuristic judge (keyword overlap with evidence, template checks, length).
Both return the same JSON schema so agreement analysis is apples-to-apples.
"""
import json
import re


def heuristic_judge(customer, historical_evidence, response):
    r = response.lower()
    c = customer.lower()
    ev_words = set(re.findall(r"[a-z]{4,}", historical_evidence.lower()))
    r_words = set(re.findall(r"[a-z]{4,}", r))
    overlap = len(ev_words & r_words) / max(1, len(ev_words | r_words))
    groundedness = 5 if overlap > 0.12 else 4 if overlap > 0.07 else 3 if overlap > 0.03 else 2
    # penalize invented concrete promises
    bad = ["refunded", "credited $", "i have rebooked", "your new flight is",
           "money sent", "guarantee"]
    correctness = 3
    if any(b in r for b in bad):
        correctness = 2
    elif overlap > 0.05 and "please share" in r:
        correctness = 4
    # --- intent-action mismatch detection (creates real variance) ---
    # Customer asks to cancel/change/rebook but reply only offers status lookup
    change_cues = ["cancel reservation", "cancel my", "change my flight", "rebook",
                   "reschedule", "standby", "switch from"]
    status_reply = ("look up the latest status" in r or "confirm the flight number and date" in r)
    loyalty_reply = ("aadvantage number" in r and "which activity is missing" in r)
    if any(k in c for k in change_cues) and status_reply:
        correctness = 2
    # Customer asks status/time but reply asks for loyalty account
    status_cues = ["what time", "on time", "status of", "where is inbound", "will aa", "does aa"]
    if any(k in c for k in status_cues) and loyalty_reply:
        correctness = 1
    # Seat complaint answered with status lookup
    if ("seat" in c or "recline" in c or "legroom" in c or "entertainment" in c) and status_reply:
        correctness = 2
    # Discrimination / legal complaint answered with rebooking template
    if any(k in c for k in ["discriminat", "lawyer", "lawsuit", "sue"]) and "rebooking options" in r:
        correctness = 1
    # Delay/cancel answered with website-troubleshooting template
    if any(k in c for k in ["cancelled", "canceled", "delayed", "missed connection"]) and "screenshot" in r:
        correctness = 2
    # High-risk underplay: medical/safety/fraud gets generic template without urgency
    risky = ["medication", "medical emergency", "discriminat", "lawyer", "fraud",
             "police", "safety issue", "stranded"]
    completeness = 4 if len(r.split()) > 25 else 3
    if any(k in c for k in risky):
        completeness = min(completeness, 2)
        correctness = min(correctness, 3)
        if correctness == 4:
            correctness = 3
    helpfulness = 4 if ("please share" in r or "confirmation" in r or "record locator" in r) else 3
    if correctness <= 2:
        helpfulness = min(helpfulness, 2)
    tone = 5 if any(w in r for w in ["sorry", "thanks", "happy to help"]) else 4
    if len(r.split()) < 10:
        helpfulness = 2
    overall = round((correctness + groundedness + helpfulness + tone + completeness) / 5)
    return {"correctness": correctness, "groundedness": groundedness,
            "helpfulness": helpfulness, "tone": tone, "completeness": completeness,
            "overall": overall, "reason": f"heuristic: overlap={overlap:.3f}"}


def judge_response(customer, historical_evidence, response):
    from src.config import GEMINI_API_KEY, GEMINI_MODEL
    if not GEMINI_API_KEY:
        return heuristic_judge(customer, historical_evidence, response)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""You are evaluating an AI customer-support response.
CUSTOMER: {customer}
HISTORICAL EVIDENCE: {historical_evidence}
AI RESPONSE: {response}
Score 1-5: correctness, groundedness, helpfulness, tone, completeness.
Penalize unsupported claims, invented policies, contradictions. Reward concise actionable replies.
Return JSON only with correctness/groundedness/helpfulness/tone/completeness/overall/reason."""
        resp = model.generate_content(prompt)
        text = resp.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        j = heuristic_judge(customer, historical_evidence, response)
        j["reason"] += f" (LLM fallback: {e})"
        return j
