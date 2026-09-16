"""Response generator: grounded drafting.

Default (offline): template that reuses the top historical agent response
pattern + intent-specific next step. Never invents refunds/credits/actions.
Optional (GEMINI_API_KEY): LLM rewrite grounded in retrieved cases with strict rules.
"""
import json

INTENT_NEXT_STEPS = {
    "flight_delay_cancel": "I'm pulling up rebooking options for you — please share your record locator and whether you can accept an alternate connection.",
    "baggage_issue": "I'll help trace the bag — please share your baggage claim/file reference (e.g. DFWAA12345) and delivery address.",
    "booking_change": "I can walk through change options — please share your confirmation code and new preferred date/time.",
    "refund_compensation": "I can check refund/voucher eligibility — please share your ticket number and the disrupted flight/date.",
    "checkin_boarding": "Let's sort check-in — please share your confirmation code and what error/message you see.",
    "seat_upgrade": "I'll check seat options — please share your flight/date and preferred seat type.",
    "customer_service": "I'm sorry about that experience — please share flight/date and what happened so I can flag it properly.",
    "flight_status_info": "I can look up the latest status — please confirm the flight number and date.",
    "loyalty_program": "I'll check the account — please share the AAdvantage number and which activity is missing.",
    "website_app_issue": "Sorry for the trouble — please share what step fails (with screenshot text/error) and the browser/app version.",
}

SAFE_OPENERS = {
    "flight_delay_cancel": "I'm sorry your flight was disrupted.",
    "baggage_issue": "I'm sorry about the trouble with your bag.",
    "booking_change": "Happy to help with that change.",
    "refund_compensation": "I understand you want an update on your refund/credit.",
    "checkin_boarding": "Let's get you checked in.",
    "seat_upgrade": "Thanks for flagging the seat issue.",
    "customer_service": "I'm sorry about that experience.",
    "flight_status_info": "Thanks for reaching out.",
    "loyalty_program": "Thanks for your loyalty — let's sort the account issue.",
    "website_app_issue": "Sorry the site/app gave you trouble.",
}


def template_reply(customer_message, intent, historical_cases):
    opener = SAFE_OPENERS.get(intent, "Thanks for reaching out.")
    step = INTENT_NEXT_STEPS.get(intent, "Please share your confirmation code so I can look into this.")
    evidence_summary = ""
    grounded_example = ""
    if historical_cases:
        top = historical_cases[0]["record"]
        grounded_example = top.get("agent_response", "")[:220]
        evidence_summary = f"Top match (score {historical_cases[0]['score']:.2f}): '{top.get('customer_message','')[:120]}...'"
    reply = (
        f"{opener} {step} "
        f"For reference, in similar cases we've typically responded like: \"{grounded_example}\" "
        f"If you'd like, I can escalate this to a specialist."
        if grounded_example else f"{opener} {step}"
    )
    return {"reply": reply.strip(), "grounded": bool(historical_cases),
            "evidence_summary": evidence_summary or "no historical evidence"}


def generate_reply(customer_message, intent, historical_cases):
    from src.config import GEMINI_API_KEY, GEMINI_MODEL
    if not GEMINI_API_KEY:
        return template_reply(customer_message, intent, historical_cases)
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        evidence = "\n".join(
            f"CASE {i+1} (score {c['score']:.2f})\nCustomer: {c['record'].get('customer_message','')}\n"
            f"Historical response: {c['record'].get('agent_response','')}"
            for i, c in enumerate(historical_cases[:5]))
        prompt = f"""You are an AmericanAir support drafter. Draft a concise professional reply.
Customer: {customer_message}
Intent: {intent}
Historical cases:
{evidence}
STRICT RULES: ground in evidence; do not invent refunds/credits/policies/actions; do not claim action completed; if evidence weak say escalation is appropriate. Return JSON with reply/grounded/evidence_summary."""
        resp = model.generate_content(prompt)
        text = resp.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        r = template_reply(customer_message, intent, historical_cases)
        r["evidence_summary"] += f" (LLM fallback: {e})"
        return r
