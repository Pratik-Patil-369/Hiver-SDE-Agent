# Technical Evaluation Report: Grounded Customer Support AI Agent

**Candidate:** Pratik Patil  
**Target Brand:** AmericanAir (`@AmericanAir`)  
**Assignment:** Hiver SDE Intern Take-Home Project  
**Repository:** [https://github.com/Pratik-Patil-369/Hiver-SDE-Agent](https://github.com/Pratik-Patil-369/Hiver-SDE-Agent)  

---

## 1. Executive Summary

This report evaluates an end-to-end AI support agent developed for **AmericanAir** on Twitter customer service data (`thoughtvector/customer-support-on-twitter`). The system:
1. Classifies incoming user messages into 10 mutually-distinguishable, data-derived customer intents.
2. Retrieves relevant historical brand resolutions to ground draft replies without hallucinating policies or false promises.
3. Decides whether to `AUTO-HANDLE` or route to `HUMAN` review using deterministic safety rules with transparent, auditable reasons.

The system was evaluated against two baseline models on an independently authored, 200-sample hand-labeled golden evaluation set. Our agent achieved a **Macro-F1 of 0.864** on intent routing (surpassing the majority baseline of **0.017**), a **1.000 recall** on safety-critical escalation, an automated response quality score of **3.95 / 5.0**, and demonstrated moderate human-judge agreement (**Pearson $r = 0.57$, Quadratic Weighted $\kappa = 0.23$**).

---

## 2. Problem Framing

### What "Good" Means for AmericanAir
Operating on Twitter requires balancing rapid response times with severe brand and safety liabilities. For American Airlines:
- **Accuracy over Coverage**: Misclassifying a flight cancellation as a simple status inquiry produces frustrated passengers and missed connections.
- **Strictly Grounded Responses**: Draft replies must mirror genuine historical resolutions (e.g., prompting for direct messages with record locators) and must *never* invent refunds, travel credits, or claim an action has been completed.
- **Fail-Safe Escalation**: Any hint of physical danger, medical crisis, legal action, financial fraud, or multi-faceted rants must immediately route to human specialists.

### Deliberate Non-Goals (What We Chose NOT to Build)
- **No Orchestration Overhead**: We avoided LangGraph, AutoGen, and complex microservice architectures. A predictable linear pipeline is faster, cheaper, and vastly easier to debug during live code reviews.
- **No Black-Box LLM Escalation**: Relying solely on prompt "vibes" to escalate cases leads to unexplainable and uncalibrated routing. Our escalation engine uses explicit deterministic thresholds.
- **No Full 3M-Tweet Blocking Dependency**: A deterministic 500-conversation knowledge base and 200-sample golden set enable instant reproduction in $<10$ seconds while preserving a clean full-dataset ingestion pipeline for production.

---

## 3. System Architecture

The pipeline consists of four modular, decoupled components:

```
[Incoming Customer Message]
           │
           ├─────────────────────────────┐
           ▼                             ▼
  [Intent Classifier]            [Retrieval Engine]
  - TF-IDF + LogisticRegression  - Sentence Transformers / TF-IDF
  - Gemini 2.5 Flash (Optional)  - FAISS / NearestNeighbors
  - Output: Intent + Conf        - Output: Top-5 Historical Cases
           │                             │
           └──────────────┬──────────────┘
                          ▼
             [Grounded Response Generator]
             - Cites historical agent resolutions verbatim
             - Zero hallucinated refunds/commitments
                          │
                          ▼
             [Deterministic Escalation Engine]
             - Confidence Threshold (< 0.55)
             - Similarity Threshold (< 0.15)
             - Risk Keyword Scanner (legal, safety, medical, fraud)
             - Query Length Heuristic (> 60 words)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      [AUTO-HANDLE]             [HUMAN ESCALATION]
                                (With stated reason)
```

---

## 4. Dataset and Sampling Strategy

- **Source Dataset**: Kaggle's *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`), containing ~3M tweets with `tweet_id`, `author_id`, `inbound`, `text`, and thread linkage fields.
- **Brand Selection**: **AmericanAir** was selected due to having the largest tweet volume, the richest diversity of operational disruptions (weather delays, lost bags, rebookings), and highly consistent resolution patterns (*"Please DM your 6-letter record locator"*).
- **Knowledge Base (KB)**: 500 curated, deduplicated historical AmericanAir interactions in `data/processed/conversations.csv`.
- **Golden Evaluation Set**: 200 hand-labeled examples in `data/golden/golden_set.csv`.
  - **Stratification**: 18 examples per intent across 10 intents ($18 \times 10 = 180$) + 20 high-risk escalation edge cases ($10\%$ base escalation rate).
  - **Leakage Prevention**: Golden examples use slot variations and syntactic structures disjoint from the 500 KB conversations, ensuring zero exact-match leakage.

---

## 5. Intent Taxonomy

Through clustering and frequency analysis of AmericanAir interactions, we defined 10 mutually-distinguishable intents:

1. `flight_delay_cancel`: Flight cancellations, mechanical delays, missed connections, diversion rebooking.
2. `baggage_issue`: Lost luggage, damaged bags, carousel delays, claim references.
3. `booking_change`: Voluntary itinerary changes, standby status, date corrections.
4. `refund_compensation`: Claims for monetary refunds, vouchers, expense reimbursement.
5. `checkin_boarding`: Mobile boarding pass failures, kiosk rejections, gate changes.
6. `seat_upgrade`: Legroom complaints, involuntary seat reassignment, first-class upgrades.
7. `customer_service`: Unhelpful agents, long telephone hold times, counter friction.
8. `flight_status_info`: Schedule queries, departure times, inbound aircraft tracking.
9. `loyalty_program`: Missing AAdvantage miles, tier status issues, login errors.
10. `website_app_issue`: App checkout crashes, payment errors, session timeouts.

---

## 6. Historical Retrieval

The retriever identifies how AmericanAir previously resolved similar customer problems:
- **Vector Representation**: L2-normalized TF-IDF vectorizer (with sentence-transformer fallback).
- **Search Backend**: FAISS `IndexFlatIP` (with automatic fallback to `sklearn.metrics.pairwise.cosine_similarity`).
- **Top-k Selection**: Retrieves the top $k=5$ closest historical interactions to inform the response generator.

---

## 7. Response Generation

To eliminate model hallucinations, the response drafter enforces strict operational guardrails:
- **Offline / Default Mode**: Employs intent-tailored safe openers and standardized action steps while quoting the top historical agent reply verbatim.
- **LLM Mode (Gemini 2.5 Flash)**: Conditioned on the retrieved historical cases with strict system instructions: *Never invent policies; never promise credits or refunds; never claim an action has been taken; ask for verification details (record locator, claim tag).*

---

## 8. Escalation Policy

Customer support escalation cannot rely on ambiguous model confidence alone. Our escalation engine executes four deterministic rules:
1. **Low Intent Confidence**: If $P(\text{intent}) < 0.55$, flag `HUMAN`.
2. **Weak Historical Similarity**: If top retrieval cosine similarity $< 0.15$, flag `HUMAN`.
3. **High-Risk Term Matching**: If the message contains high-liability keywords (`lawyer`, `legal`, `lawsuit`, `fraud`, `scam`, `police`, `medical`, `emergency`, `safety`, `threat`, `discrimination`), immediately flag `HUMAN`.
4. **Complexity / Length Guard**: If query length exceeds 60 words (indicating multi-part rants), flag `HUMAN`.

---

## 9. Evaluation Methodology

### Baseline Models
1. **Majority Baseline (Trivial)**: Predicts the most frequent training class for every test query.
2. **TF-IDF + Logistic Regression (Simple)**: Trained exclusively on weak keyword rules over the KB (zero exposure to the golden set).

### LLM-as-a-Judge Rubric
Each generated reply is evaluated across 5 dimensions on a 1–5 Likert scale:
- **Correctness**: Alignment with customer problem; severe penalty for conflicting actions.
- **Groundedness**: Support from historical evidence; penalizes ungrounded promises.
- **Helpfulness**: Actionable guidance (requesting record locator, claim code).
- **Tone**: Professionalism and empathy.
- **Completeness**: Thoroughness in addressing all customer questions.

### Human-Judge Agreement
A subset of 40 replies was scored by human review using the exact same 5-dimension rubric. Agreement was evaluated using **Pearson correlation ($r$)** and **Quadratic Weighted Kappa ($\kappa$)**.

---

## 10. Benchmark Results

### Intent Classification
| Model / Pipeline | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| **Majority Classifier** | 0.090 | 0.009 | 0.100 | **0.017** |
| **TF-IDF + Logistic Regression** | 0.860 | 0.882 | 0.869 | **0.864** |
| **AI Support Agent** | 0.860 | 0.882 | 0.869 | **0.864** |

### Safety Escalation Performance
- **Recall**: **1.000** (20/20 critical cases caught — zero misses on safety, legal, medical, or fraud).
- **Precision**: **0.488** (21 false alarms out of 41 total human escalations).
- **F1 Score**: **0.656**.

### Response Quality & Judge Agreement
- **Overall Judge Score**: **3.95 / 5.0** (Correctness: 3.57, Groundedness: 5.00, Helpfulness: 3.71, Tone: 4.99, Completeness: 3.80).
- **Human vs. Judge Agreement ($n=40$)**:
  - Pearson $r = \mathbf{0.57}$
  - Quadratic Weighted $\kappa = \mathbf{0.23}$
  - Human Mean = $3.53$ vs. Judge Mean = $3.93$ (Judge exhibits mild leniency bias).

---

## 11. Top 5 Failure Modes & Hypotheses

From systematic review of `results/failures.csv`:

1. **The Disruption Triangle (`flight_delay_cancel` $\leftrightarrow$ `booking_change` $\leftrightarrow$ `flight_status_info`)**:
   - *Example*: User attempting to cancel a ticket due to a delay is classified as status lookup.
   - *Cause*: High vocabulary overlap across delay and rebooking contexts.
2. **Flight Number Sparsity**:
   - *Example*: Alphanumeric flight numbers (e.g., `AA334`) confused with AAdvantage account numbers.
   - *Cause*: Bag-of-words tokens lacking entity extraction structure.
3. **Implicit Seat Complaints**:
   - *Example*: *"Moved from window to middle without asking"* misclassified.
   - *Cause*: Absence of explicit keyword `"seat"`.
4. **Draft Reply Tone Underplay on Escalation**:
   - *Example*: High-risk legal/medical emergencies correctly flag `HUMAN`, but the drafted reply still outputs an auto-response.
   - *Cause*: Reply drafter and escalation engine run concurrently without mutual suppression.
5. **Over-Escalation on Brief Queries**:
   - *Example*: Short, valid inquiries (e.g., *"Status AA45?"*) yield low cosine scores ($<0.15$), triggering unnecessary human escalation.

---

## 12. What is Misleading About My Headline Number? (Mandatory Section)

As engineers, transparency matters more than flattering metrics:
- **Macro-F1 0.864 Overstates Production Readiness**: Real Twitter traffic is heavily degraded by typos, slang, sarcasm, and multi-tweet fragmentation not fully captured in the golden set.
- **The Agent Intent Score Equals the Baseline by Construction**: Both use the same TF-IDF backbone offline; the agent's actual differentiation is in grounded response drafting and escalation safety.
- **1.000 Escalation Recall Masks Human Operational Costs**: Achieving zero safety misses came at the cost of a 48.8% precision, meaning approximately half of all human reviews are false alarms.
- **Groundedness 5.0 is an Artifact of Quoting**: Direct quotation of historical responses produces near-perfect lexical overlap with evidence, artificially inflating groundedness scores.
- **Statistical Power ($n=200$)**: A 200-sample test set carries an empirical margin of error of $\pm 5\text{--}7\%$ on individual intent classes.

---

## 13. What I Would Do With One More Week

1. **Full Kaggle Ingestion**: Process the full 3M-tweet dataset to quantify production domain shift.
2. **Suppression Switch**: Implement an immediate draft-kill mechanism so escalated queries return an acknowledgment rather than an auto-drafted reply.
3. **Dynamic Escalation Thresholds**: Calibrate similarity thresholds dynamically based on query token length.
4. **Retrieval Recall@5 Benchmark**: Evaluate vector retrieval with human relevance labels.
5. **Multi-Rater Agreement**: Involve a second annotator to measure Human-to-Human $\kappa$ alongside the LLM Judge.

---

## 14. Conclusion & Decision Log

The system demonstrates that dependable customer support AI is achieved not through uncontrolled generative freedom, but through **defensive architecture, explicit safety guardrails, and rigorous evaluation**. 

For the complete log of 15 non-obvious engineering decisions, please see [`decision_log.md`](decision_log.md).

