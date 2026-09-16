# Golden-set labeling methodology (200 examples)

- **Brand**: AmericanAir (largest brand in twcs by author_id volume; diverse delay/baggage/refund traffic).
- **Source**: template variants mirroring real AmericanAir phrasing in twcs ("flight AA123 cancelled", "bag never arrived", "on hold 2 hours"), with disjoint slot fills from the retrieval KB so no exact-match leakage.
- **Sampling**: stratified — 18 per intent × 10 intents = 180 + 20 high-risk escalation cases (legal/medical/safety/fraud/multi-part) = 200. Shuffled, seed 42.
- **Labels**: `true_intent` per src/intents.py definitions (primary customer need; keywords alone insufficient, e.g. "where is my refund" → refund_compensation not booking_change). `true_escalation=true` only when a human should handle: legal threat, safety/medical, fraud/police, discrimination, or >60-word multi-request messages.
- **Verification**: all 200 reviewed by author against definitions; 40-item subset double-scored for LLM-judge agreement (see data/golden/human_scores.csv).
- **Limitation**: seed phrasing is cleaner than raw Twitter noise (typos, sarcasm under-represented) — see README "What is misleading".
