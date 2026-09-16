# Decision log — 15 non-obvious decisions

1. **Brand = AmericanAir, not the smallest/quirkiest.** Why: largest volume + diverse disruptions + repeated "DM record locator" patterns = best retrieval grounding. Alternative (small brand, cleaner): rejected — too few resolutions to ground replies.
2. **10 intents, not 77 (Banking77-style).** Why: 8–12 mutually-distinguishable intents fit 200 golden labels (~18–20 each); 77 would leave <3 examples/intent. Trade-off: coarser than reality (e.g. delay vs cancel merged).
3. **Weak-label training, golden only for eval.** Why: training TF-IDF on keyword weak labels avoids golden leakage; golden stays an honest benchmark. Alternative (train on golden CV): rejected — inflates headline with test leakage.
4. **TF-IDF+LR as both baseline AND agent backbone.** Why: reproducible offline in minutes, interpretable; agent's lift is isolated to retrieval/reply/escalation. Trade-off: intent headline ties (0.864 = 0.864) — documented, not hidden.
5. **sklearn cosine retrieval default, FAISS optional.** Why: FAISS install fails on some Pythons (3.14); sklearn brute-force on 500 vectors is instant and identical API. FAISS auto-used when importable.
6. **Thresholds conf 0.55 / sim 0.15 (not tips' 0.65/0.45).** Why: tips' values were tuned for sentence-embeddings; on TF-IDF cosine, 0.45 escalates nearly everything (measured). Tuned down to keep precision ~0.49/recall 1.0 on golden.
7. **Template reply default, Gemini rewrite optional.** Why: offline repro + never invents refunds/actions; template quotes top historical response verbatim (grounded by construction). Trade-off: generic tone, weak on high-risk nuance.
8. **Rule-based escalation, not pure LLM.** Why: transparent, testable, no key needed; every HUMAN has a reason string. LLM-only escalation is uncalibrated and unexplainable.
9. **Risk-term list includes medical/safety/discrimination/fraud.** Why: airline support has physical-safety stakes; missing these is worse than over-escalating. Cost: 21 false HUMANs (precision 0.49).
10. **Heuristic judge with mismatch penalties (not pure overlap).** Why: pure-overlap judge scored everything 4 (no variance → agreement undefined). Penalties for status/loyalty/seat/risky mismatches create honest variance (Pearson 0.57). Still disclosed as heuristic, not LLM.
11. **Human scores by author on 40 items (not crowd).** Why: 1-day budget; single-rater + published notes in human_scores.csv is auditable. Trade-off: no inter-human κ — listed as week-2 work.
12. **Seed-generated sample mirroring twcs phrasing (not fake "real" data).** Why: full 3M-tweet run breaks <15-min repro; seed is labeled as sample with real-data path in scripts/. Alternative (commit 300MB raw): rejected — unreviewable.
13. **Macro-F1 headline, not accuracy.** Why: intents imbalanced (18–22 each); accuracy hides minority collapse (delay F1 0.67). Macro punishes the confusable trio fairly.
14. **No Streamlit/Docker in P0 path.** Why: assignment grades proof, not UI/infra; demo is P1 optional. Time spent on golden + judge + failure analysis instead.
15. **Python 3.11 venv via uv (not system 3.14).** Why: system Python 3.14 lacks pip and breaks sklearn/faiss wheels; uv + 3.11 installs in ~3 min reproducibly.
