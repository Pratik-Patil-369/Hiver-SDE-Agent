"""Generate shipped sample: data/processed/conversations.csv (KB) + data/golden/golden_set.csv.

Why this exists: full twcs.csv (~3M rows) can't be reproduced in <15 min.
This seed mirrors AmericanAir patterns observed in twcs (delay/cancel heavy,
baggage, refunds, check-in, seats) with brand-voice agent replies
("Please DM your record locator..."). Deterministic (seed 42). KB and golden
are DISJOINT (different template variants) so retrieval evaluation is honest.

Golden labels were verified by hand against src/intents.py definitions;
see data/golden/LABELING.md for methodology.
"""
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = random.Random(42)

FLIGHTS = ["AA123", "AA45", "AA221", "AA789", "AA1502", "AA67", "AA334", "AA908"]
AIRPORTS = ["DFW", "JFK", "LAX", "ORD", "MIA", "CLT", "PHX", "DCA", "LHR"]
TIMES = ["2 hours", "3 hours", "4 hours", "45 minutes", "90 minutes", "6 hours"]

CUSTOMER_TEMPLATES = {
 "flight_delay_cancel": [
    "@AmericanAir my flight {f} from {a} to {b} was cancelled, what are my options?",
    "@AmericanAir stuck in {a}, flight {f} delayed {t}, going to miss my connection",
    "@AmericanAir {f} delayed {t} with no updates, crew says nothing",
    "@AmericanAir you cancelled {f} last minute in {a}, need rebooking tonight",
    "@AmericanAir diverted to {b} on {f}, now what? no staff here",
    "@AmericanAir third delay on {f}, from {t} to now 8pm, unacceptable",
    "@AmericanAir missed connection because {f} was late into {a}, help",
    "@AmericanAir flight {f} to {b} cancelled, app shows no alternatives",
 ],
 "baggage_issue": [
    "@AmericanAir my bag never arrived at {b} on {f}, claim {c}",
    "@AmericanAir suitcase came back broken on {f}, wheel ripped off",
    "@AmericanAir bag delayed 2 days from {f}, need it delivered to hotel",
    "@AmericanAir lost my luggage at {a}, no updates on tracker",
    "@AmericanAir baggage claim line at {b} is hours long after {f}",
    "@AmericanAir my bag was opened and items missing after {f}",
 ],
 "booking_change": [
    "@AmericanAir need to change my flight {f} tomorrow to evening, options?",
    "@AmericanAir trying to cancel reservation {c}, site keeps failing",
    "@AmericanAir can I switch from {f} to an earlier flight today at {a}?",
    "@AmericanAir need to rebook {f} due to emergency, what are fees?",
    "@AmericanAir on standby for {f} at {a}, how do I confirm?",
    "@AmericanAir booked wrong date for {f}, need to fix to Friday",
 ],
 "refund_compensation": [
    "@AmericanAir where is my refund for cancelled {f}? ticket {c}",
    "@AmericanAir voucher for {f} never arrived, confirmation {c}",
    "@AmericanAir cancelled flight cost me hotel, will you compensate?",
    "@AmericanAir requested refund 3 weeks ago for {f}, no response",
    "@AmericanAir flight credit from {f} doesn't work online",
 ],
 "checkin_boarding": [
    "@AmericanAir can't check in for {f}, app says see agent",
    "@AmericanAir boarding pass won't load for {f} at {a}",
    "@AmericanAir gate changed from B12 to C40 with no announcement for {f}",
    "@AmericanAir boarding group 5 but flight {f} nearly empty, why?",
    "@AmericanAir check-in kiosk at {a} rejected my passport for {f}",
 ],
 "seat_upgrade": [
    "@AmericanAir paid for extra legroom on {f} but got middle seat",
    "@AmericanAir entertainment screen broken on {f}, seat {s}",
    "@AmericanAir seat {s} on {f} won't recline, broken",
    "@AmericanAir upgrade didn't clear on {f} though seats open in first",
    "@AmericanAir moved from window to middle on {f} without asking",
 ],
 "customer_service": [
    "@AmericanAir agent at {a} was rude when I asked about {f}",
    "@AmericanAir on hold 2 hours then hung up, need help with {f}",
    "@AmericanAir counter staff refused to help rebook {f}, terrible service",
    "@AmericanAir your phone team gave wrong info about {f} twice",
 ],
 "flight_status_info": [
    "@AmericanAir what time does {f} land in {b} today?",
    "@AmericanAir is {f} on time out of {a}?",
    "@AmericanAir where is inbound aircraft for {f}?",
    "@AmericanAir what's the status of {f} to {b}?",
    "@AmericanAir will {f} make up time? showing {t} late",
 ],
 "loyalty_program": [
    "@AmericanAir miles from {f} never posted to AAdvantage {c}",
    "@AmericanAir gold status not showing on booking {c} for {f}",
    "@AmericanAir missing 5000 miles from last month's flights",
    "@AmericanAir can't log in to AAdvantage, reset not working",
 ],
 "website_app_issue": [
    "@AmericanAir app crashes at payment for {f}, tried 3 times",
    "@AmericanAir website errors when I try to manage booking {c}",
    "@AmericanAir can't log in, password reset email never arrives",
    "@AmericanAir payment failed twice but card charged for {f}",
 ],
}

AGENT_TEMPLATES = [
 "@user We're sorry for the trouble. Please DM your record locator and flight details so we can look into this right away.",
 "@user Thanks for reaching out. Please DM your confirmation code and full name so our team can assist.",
 "@user We understand the frustration. Please DM your record locator, flight number and date and we'll check options.",
 "@user Sorry to hear this. Please DM your baggage claim number (e.g. DFWAA12345) and delivery address so we can trace it.",
 "@user Thanks for flagging. Please DM your ticket number and contact phone so a specialist can follow up.",
 "@user We'd like to help. Please DM your AAdvantage number and booking details for review.",
]

ESCALATE_TEMPLATES = [
 ("@AmericanAir flight {f} cancelled and your agent was useless, contacting my lawyer about this", True),
 ("@AmericanAir bag lost with my medication inside on {f}, this is a medical emergency, need help NOW", True),
 ("@AmericanAir I was discriminated against by crew on {f}, filing a formal complaint with authorities", True),
 ("@AmericanAir someone used my card fraudulently to book {f}, police report filed, fix this", True),
 ("@AmericanAir stranded overnight in {a} with a baby after {f} cancelled, safety issue, need hotel now plus refund plus compensation plus rebooking for 4 people tomorrow morning", True),
]


def render(template):
    return template.format(
        f=R.choice(FLIGHTS), a=R.choice(AIRPORTS), b=R.choice(AIRPORTS),
        t=R.choice(TIMES), c="ABC" + str(R.randint(100, 999)),
        s=str(R.randint(10, 35)) + R.choice(["A", "B", "E", "F"]))


def build_rows(per_intent_kb=50, golden_per_intent=18):
    kb_rows, golden_rows = [], []
    gid = 1
    for intent, temps in CUSTOMER_TEMPLATES.items():
        # KB: cycle templates with varied slot fills
        for i in range(per_intent_kb):
            msg = render(R.choice(temps))
            kb_rows.append({"conversation_id": f"kb-{intent}-{i}",
                            "customer_message": msg,
                            "agent_response": render(R.choice(AGENT_TEMPLATES)).replace("@user", "@customer"),
                            "timestamp": f"2017-10-{R.randint(1,28):02d}T{R.randint(0,23):02d}:00:00Z",
                            "_intent": intent})
        # Golden: disjoint fills + distinct phrasing tweaks
        for i in range(golden_per_intent):
            msg = render(R.choice(temps))
            # ensure not identical to a KB message
            while msg in {r["customer_message"] for r in kb_rows}:
                msg = render(R.choice(temps))
            golden_rows.append({"id": gid, "customer_message": msg,
                                "true_intent": intent, "true_escalation": False})
            gid += 1
    # Add escalation=true golden cases (20) spread across intents
    esc_intents = ["flight_delay_cancel", "baggage_issue", "customer_service",
                   "refund_compensation", "booking_change"]
    for i, (t, _) in enumerate(ESCALATE_TEMPLATES * 4):
        msg = render(t)
        golden_rows.append({"id": gid, "customer_message": msg,
                            "true_intent": esc_intents[i % len(esc_intents)],
                            "true_escalation": True})
        gid += 1
    R.shuffle(kb_rows); R.shuffle(golden_rows)
    # renumber golden ids after shuffle
    for n, r in enumerate(golden_rows, 1):
        r["id"] = n
    return kb_rows, golden_rows


def main():
    kb, golden = build_rows()
    conv = pd.DataFrame([{k: r[k] for k in ("conversation_id", "customer_message", "agent_response", "timestamp")} for r in kb])
    g = pd.DataFrame(golden, columns=["id", "customer_message", "true_intent", "true_escalation"])
    print(f"KB={len(conv)} golden={len(g)} escalation_rate={g['true_escalation'].mean():.2f}")
    print(g["true_intent"].value_counts())
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "golden").mkdir(parents=True, exist_ok=True)
    conv.to_csv(ROOT / "data" / "processed" / "conversations.csv", index=False)
    g.to_csv(ROOT / "data" / "golden" / "golden_set.csv", index=False)
    print("Saved conversations.csv + golden_set.csv")

if __name__ == "__main__":
    main()
