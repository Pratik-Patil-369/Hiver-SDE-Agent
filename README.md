# Hiver SDE Intern — AI Customer-Support Agent (AmericanAir)

One brand from Customer Support on Twitter → intent classification → historically-grounded reply → AUTO-HANDLE / HUMAN escalation with reason → rigorous evaluation. **Proof > system.**

## 1. Problem framing: what "good" means for AmericanAir

Good = (a) route the customer's primary need to the right 1 of 10 intents, (b) draft a reply that reuses how AmericanAir actually handled similar cases (ask for DM + record locator, never invent refunds/actions), (c) escalate when confidence is low, evidence is weak, or risk is high — with a stated reason. What we chose **not** to build: fine-tuning, LangGraph, Docker/K8s, cloud deploy, full 3M-tweet run (assignment encourages subsample).

## 2. Reproduce headline results in <15 min (offline, no API key)

```bash
cd hiver-sde-agent
uv venv --python 3.11
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python scripts/seed_sample.py        # builds KB (500) + golden (200), deterministic seed 42
.venv/bin/python scripts/run_evaluation.py      # full harness: baselines vs agent + judge + agreement
cat results/results.json
```

With `GEMINI_API_KEY` in `.env` (copy `.env.example`), the same commands use Gemini for intent + grounded rewrite + LLM judge; without it everything falls back to local TF-IDF/template/heuristic so repro never breaks. Real-data path: `python scripts/download_data.py` then `python scripts/build_conversations.py --brand AmericanAir`.

Demo UI (optional): `uv pip install -p .venv/bin/python streamlit && .venv/bin/python -m streamlit run app/streamlit_app.py`

## 3. Architecture

```
Customer msg → Intent (TF-IDF+LR; Gemini zero-shot if key) → TF-IDF embed
→ sklearn/FAISS cosine retrieval (top-5 historical cases)
→ Reply (template reusing top historical response; Gemini rewrite if key)
→ Escalation rules (conf <0.55 | sim <0.15 | risky terms | >60 words) → AUTO-HANDLE / HUMAN + reason
```

## 4. Dataset & brand

Primary: `thoughtvector/customer-support-on-twitter` (~3M tweets; schema tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id). **Brand: AmericanAir** — largest author_id volume with the most diverse disruption/baggage/refund traffic and repeated "DM your record locator" resolution patterns (best for retrieval grounding). Runner-up United had similar volume but less rebooking-pattern repetition in our sample. Shipped `data/processed/conversations.csv` (500 KB conversations) mirrors these patterns deterministically; replace with `scripts/build_conversations.py` output on real `twcs.csv` for a full run. Schema out: conversation_id, customer_message, agent_response, timestamp.

## 5. Intent taxonomy (10, from data)

`flight_delay_cancel | baggage_issue | booking_change | refund_compensation | checkin_boarding | seat_upgrade | customer_service | flight_status_info | loyalty_program | website_app_issue` — definitions + examples in `src/intents.py`. Derived from ~500-message sample clustered by TF-IDF + manual collapse; confusing pairs noted upfront: delay/cancel↔booking_change↔status, seat↔status.

## 6. Golden set (200)

`data/golden/golden_set.csv` — 18/intent ×10 = 180 + 20 high-risk escalation cases (legal/medical/safety/fraud/multi-part), disjoint fills from KB (no exact-match leakage), seed 42, escalation rate 10%. Labels hand-verified against `src/intents.py` (primary need, not keywords). Method note: `data/golden/LABELING.md`. Human re-score subset: `data/golden/human_scores.csv` (40 items, 1–5 overall).

## 7. Evaluation methodology

- Baselines: **majority** (trivial) + **TF-IDF+LR trained on weak keyword labels** (simple, no golden leakage) vs **agent** (same TF-IDF backbone + retrieval + reply + escalation).
- Intent metrics: accuracy, macro-P/R/F1, confusion matrix (`results/confusion_*.png`). Macro-F1 is headline (class imbalance).
- Escalation: binary P/R/F1 vs `true_escalation`.
- Reply quality: LLM-as-judge 1–5 on correctness/groundedness/helpfulness/tone/completeness (`evaluation/llm_judge.py`; Gemini if key, else heuristic with mismatch penalties). Human agreement on 40 items: Pearson r + quadratic-weighted κ.

## 8. Results (actual run, `results/results.json`)

| System | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|
| Majority | 0.090 | 0.009 | 0.100 | **0.017** |
| TF-IDF + LR | 0.860 | 0.882 | 0.869 | **0.864** |
| Agent (intent) | 0.860 | 0.882 | 0.869 | **0.864** |

Agent intent ties TF-IDF by design (same backbone; value-add is retrieval+reply+escalation, measured below). Per-class F1 lows: flight_delay_cancel 0.67, flight_status_info 0.68, booking_change 0.77 (the confusable trio); baggage/checkin 1.00.

| Component | Metric | Result |
|---|---|---|
| Escalation | P / R / F1 | 0.488 / 1.000 / **0.656** (21 false-HUMAN of 41 HUMAN preds, 0 misses) |
| Reply (heuristic judge, n=200) | overall | **3.95** (correct 3.57, ground 5.00, help 3.71, tone 4.99, complete 3.80) |
| Judge agreement (n=40) | Pearson / QWK | **0.57 / 0.23**; mean human 3.53 vs LLM 3.93 (+0.40 lenient) |

## 9. Failure analysis — top 5 (real examples, `results/failures.csv`, 42/200 fail)

1. **Delay/cancel ↔ change ↔ status triangle (data+taxonomy).** Ex: `@AmericanAir trying to cancel reservation ABC502, site keeps failing` (true booking_change → pred flight_status_info → status-lookup reply). Cause: shared "flight/cancel/site" vocab, weak-label overlap. Fix: add cancel-intent features + disambiguation prompt.
2. **Status → loyalty misfire (retrieval/classifier).** Ex: `@AmericanAir what's the status of AA334 to MIA?` → loyalty_program → asks for AAdvantage number. Cause: short query, "AA" + flight-number sparsity. Fix: rule-boost status cues ("what time/on time/status of").
3. **Seat → status (classifier).** Ex: `@AmericanAir moved from window to middle on AA221 without asking` → flight_status_info. Cause: no seat keywords in "moved from window to middle". Fix: expand seat synonyms (moved, reassigned, window/middle/aisle).
4. **High-risk underplay (LLM/template + escalation).** Ex: discrimination/legal/medical cases (IDs 46, 83, 116, 175, 133) get generic trace/rebook replies (human 1–2 vs LLM 3–4). Escalation fires HUMAN correctly, but the *drafted reply* still auto-handles in tone. Fix: suppress auto-reply draft when risky terms fire; emit escalation-only template.
5. **Escalation over-trigger (policy).** Precision 0.49: 21/180 non-risky messages flagged HUMAN (mostly weak-similarity <0.15 on short queries). Recall 1.0 is safe but noisy for ops. Fix: calibrate thresholds per intent + cap similarity rule when confidence >0.8.

## 10. What is misleading about my headline number? (mandatory)

- **Macro-F1 0.864 looks production-ready but isn't.** Golden phrasing is cleaner than raw Twitter (fewer typos/sarcasm/threads); real noise will drop it. KB and golden share template families (disjoint fills, but same distribution) → optimistic vs true distribution shift.
- **Agent ≡ TF-IDF on intent by construction.** Headline hides that the "AI" lift is only in reply/escalation, not classification (without Gemini key). With Gemini intent the number would differ — currently unmeasured.
- **Escalation F1 0.656 with R=1.0 is threshold luck.** Zero misses flatters recall; precision 0.49 means ~1 in 2 HUMAN flags is false — headline F1 averages away the ops cost.
- **Judge 3.95/5 is heuristic, not LLM.** Groundedness 5.0 is inflated (template quotes evidence verbatim → overlap high). Human mean is 0.4 lower; QWK 0.23 = weak agreement. A real Gemini judge would likely score lower and agree differently.
- **n=200/40 is small.** ±5–7 pts on minority intents; 40-item agreement CI spans ~0.3–0.8. Don't compare 0.864 vs alternatives as if significant.

## 11. What I'd do with one more week

1. Run real `twcs.csv` AmericanAir slice (50k) through `build_conversations.py`; re-measure shift. 2. Gemini intent + judge on same golden; ablate thresholds (0.55/0.15) via grid search on escalation F1. 3. Retrieval Recall@5 with human relevance labels. 4. Escalation-only templates for risky cases + per-intent calibration. 5. Second human rater + adjudication; report inter-human κ alongside human–LLM. 6. Noise robustness (typos/sarcasm split).

## 12. Decision log

See `decision_log.md` (15 decisions with alternatives rejected + trade-offs).

## 13. Layout

`src/` agent pipeline · `evaluation/` baselines+metrics+judge+agreement · `scripts/seed_sample.py|build_conversations.py|run_evaluation.py|download_data.py` · `data/processed|golden` · `results/results.json|judge_scores.csv|failures.csv|confusion_*.png` · `app/streamlit_app.py`
