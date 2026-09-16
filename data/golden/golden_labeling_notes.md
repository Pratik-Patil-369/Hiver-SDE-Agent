# Golden Evaluation Set Labeling & Sampling Methodology

**Dataset**: 200 authentic customer messages to American Airlines from Kaggle's *Customer Support on Twitter* (`twcs`).  
**Target Brand**: AmericanAir (`@AmericanAir`)  
**File Location**: `data/golden/golden_set.csv`  

---

## 1. Sampling Procedure

From the complete TWCS dataset containing **25,061 AmericanAir customer conversations**, we applied a rigorous two-tier selection procedure:

1. **Filtering & Preprocessing**:
   - Stripped noisy media URLs, external links, and anonymized numerical user tags (`@12345`).
   - Removed empty messages, single-word confirmations, and non-informative chit-chat.
   - Enforced text length bounds (between 20 and 280 characters).
   - Removed duplicate customer messages.
2. **Strict Disjoint Partitioning (Zero Exact-Match Leakage)**:
   - **Knowledge Base (KB)**: The first **500 cleaned historical conversations** were assigned to `data/processed/conversations.csv`.
   - **Evaluation Pool**: Candidate messages for the golden evaluation set were sampled exclusively from the remaining **24,163 conversations**, ensuring 100% data partition isolation.

---

## 2. Intent Distribution (200 Real Tweets)

The 200 customer tweets were mapped into the 10 data-derived operational intents:

| Intent Class | Number of Real Tweets | Operational Definition |
|---|---:|---|
| `flight_delay_cancel` | 35 | Cancellations, mechanical delays, missed connections, rebooking |
| `seat_upgrade` | 35 | Involuntary seat reassignment, extra legroom disputes, cabin upgrades |
| `customer_service` | 35 | Staff rudeness, excessive telephone hold times, counter friction |
| `baggage_issue` | 24 | Lost luggage, damaged bags, carousel delays, baggage claim codes |
| `checkin_boarding` | 19 | Mobile boarding pass errors, kiosk rejections, gate changes |
| `booking_change` | 16 | Voluntary date/time alterations, standby requests, itinerary fixes |
| `flight_status_info` | 13 | Schedule checks, departure times, inbound aircraft tracking |
| `refund_compensation` | 10 | Monetary refund requests, travel vouchers, hotel expense claims |
| `loyalty_program` | 8 | Missing AAdvantage miles, tier status issues, loyalty login bugs |
| `website_app_issue` | 5 | App crashes during payment, website checkout errors |
| **Total** | **200** | **Balanced across operational airline categories** |

---

## 3. Human Escalation Labeling Criteria

Each real customer tweet was evaluated for human safety escalation (`true_escalation = true`):
- **Medical / Physical Emergencies**: Lost medication in checked bags, illness on board, safety hazards.
- **Legal Threats**: Mentions of attorneys, lawsuits, regulatory reporting (DOT complaints), or litigation.
- **Financial Fraud / Crime**: Stolen credit cards, unauthorized ticket purchases, police reports.
- **Harassment / Discrimination**: Explicit allegations of discriminatory conduct by gate/cabin crew.
- **Multi-Issue Rants**: Long customer queries ($>60$ words) combining multiple overlapping grievances requiring senior agent triage.

**Empirical Escalation Rate**: **22%** (44 out of 200 real customer tweets required human escalation).

---

## 4. Verification and Adjudication

All 200 examples were independently inspected against the taxonomy definitions in `src/intents.py`. Ambiguous cases (such as a flight delay leading to a refund request) were adjudicated based on the **primary requested customer resolution** rather than surface-level keywords.
