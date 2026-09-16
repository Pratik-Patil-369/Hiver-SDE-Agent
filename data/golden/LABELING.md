# Golden Evaluation Set Labeling Methodology (200 Real Customer Tweets)

- **Brand**: AmericanAir (`@AmericanAir`) — chosen for its rich variety of operational flight disruptions and consistent Twitter customer service resolution patterns.
- **Source**: 200 authentic customer tweets extracted from Kaggle's *Customer Support on Twitter* (`twcs`). Candidate tweets were sampled from 24,000+ interactions strictly disjoint from the 500 KB conversations (zero exact-match leakage).
- **Annotation Process**: Initial intent and escalation candidates were generated via AI-assisted pre-labeling (`scripts/create_real_golden.py`) and then systematically reviewed, cleaned, and adjudicated against the 10 operational intent definitions in `src/intents.py`.
- **Escalation Ground Truth**: `true_escalation=true` was assigned to 44/200 cases (22% base rate) representing medical/safety crises, legal/regulatory threats, financial fraud, discrimination, or long multi-part grievances.
- **Detailed Notes**: See `data/golden/golden_labeling_notes.md` for per-intent breakdown and category criteria.
