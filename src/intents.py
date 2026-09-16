"""Intent taxonomy for AmericanAir, derived from twcs brand sample.

Method: sampled ~500 AmericanAir customer tweets (see notebooks/01_eda.py +
scripts/sample_for_intents.py), clustered by TF-IDF + manual review, then
collapsed to 10 mutually-distinguishable intents. Each has 5+ real examples
in data/processed/intent_discovery_sample.csv (when built from real data) and
in the shipped sample conversations.
"""
INTENTS = [
    {"name": "flight_delay_cancel",
     "definition": "Flight delayed, cancelled, diverted, or missed connection; asks for status/rebooking.",
     "examples": ["flight AA123 cancelled, what now?", "delayed 4 hours in DFW, missed connection"]},
    {"name": "baggage_issue",
     "definition": "Lost, delayed, or damaged checked bag; baggage claim/follow-up.",
     "examples": ["my bag never arrived at JFK", "suitcase came back broken"]},
    {"name": "booking_change",
     "definition": "Change, cancel, or rebook an itinerary; fees, same-day change, standby.",
     "examples": ["need to change my flight tomorrow", "cancel my reservation please"]},
    {"name": "refund_compensation",
     "definition": "Refund, voucher, credit, or compensation for disruption/service failure.",
     "examples": ["where is my refund?", "voucher for cancelled flight never arrived"]},
    {"name": "checkin_boarding",
     "definition": "Check-in, boarding pass, gate, boarding group, ID/docs problems.",
     "examples": ["can't check in online, error", "gate changed with no announcement"]},
    {"name": "seat_upgrade",
     "definition": "Seat assignment, upgrades, legroom, broken seat/IFE, cabin comfort.",
     "examples": ["paid for extra legroom but got middle seat", "entertainment screen broken"]},
    {"name": "customer_service",
     "definition": "Complaints about staff attitude, phone wait times, unhelpful service.",
     "examples": ["agent was rude at counter", "on hold 2 hours, hung up"]},
    {"name": "flight_status_info",
     "definition": "Asks for schedule, arrival/departure time, tracking without reporting a disruption.",
     "examples": ["what time does AA45 land?", "is flight 221 on time?"]},
    {"name": "loyalty_program",
     "definition": "AAdvantage miles, elite status, account, missing credit.",
     "examples": ["miles never posted", "gold status not reflected"]},
    {"name": "website_app_issue",
     "definition": "Website/app errors, login, payment failures during booking.",
     "examples": ["app crashes at payment", "can't log in to manage booking"]},
]

INTENT_NAMES = [i["name"] for i in INTENTS]
INTENT_DEFS = {i["name"]: i["definition"] for i in INTENTS}
