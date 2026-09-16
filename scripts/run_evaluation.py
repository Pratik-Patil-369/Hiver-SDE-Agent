"""Full evaluation: baselines vs agent on golden set + escalation + LLM judge + agreement.

Usage: python scripts/run_evaluation.py [--judge-n 200] [--k 5]
Outputs: results/results.json, results/confusion_*.png, results/judge_scores.csv
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intent_classifier import train_weakly_supervised, predict_with_confidence, weak_label
from evaluation.baseline_majority import MajorityClassifier
from evaluation.baseline_tfidf import build_model
from evaluation.metrics import evaluate_intent, save_confusion_matrix
from evaluation.llm_judge import judge_response
from src.agent import Agent

p = argparse.ArgumentParser()
p.add_argument("--judge-n", type=int, default=200)
p.add_argument("--k", type=int, default=5)
a = p.parse_args()

conv = pd.read_csv(ROOT / "data" / "processed" / "conversations.csv")
golden = pd.read_csv(ROOT / "data" / "golden" / "golden_set.csv")
print(f"KB={len(conv)} golden={len(golden)} esc_rate={golden['true_escalation'].mean():.3f}")

weak = conv["customer_message"].map(weak_label)
# Baseline 1: majority
maj = MajorityClassifier(); maj.fit(weak)
maj_pred = maj.predict(golden["customer_message"].tolist())
# Baseline 2: TF-IDF trained on WEAK labels (no golden leakage)
tfidf = build_model()
tfidf.fit(conv["customer_message"], weak)
tfidf_pred = tfidf.predict(golden["customer_message"].tolist())
# Agent execution on real golden set
agent = Agent(conv, use_llm_intent=False)
agent_intents, agent_escs, replies, cases_list = [], [], [], []
top1_sims, top5_sims = [], []
failures = []

print(f"Evaluating {len(golden)} real customer queries through Agent pipeline...")
for idx, row in golden.iterrows():
    msg = row["customer_message"]
    true_intent = row["true_intent"]
    true_esc = row["true_escalation"]
    
    r = agent.handle(msg, k=a.k)
    pred_intent = r["intent"]["intent"]
    pred_esc = r["escalation"]["decision"]
    reply_text = r["reply"]["reply"]
    cases = r["historical_cases"]
    
    agent_intents.append(pred_intent)
    agent_escs.append(pred_esc)
    replies.append(reply_text)
    
    if cases:
        top1_sims.append(cases[0]["score"])
        top5_sims.append(sum(c["score"] for c in cases) / len(cases))
    else:
        top1_sims.append(0.0)
        top5_sims.append(0.0)
        
    ev = "\n".join(c["record"].get("agent_response", "") for c in cases[:3])
    cases_list.append(ev)
    
    # Record failures for failure analysis
    if pred_intent != true_intent or (true_esc and pred_esc != "HUMAN"):
        failures.append({
            "id": int(row["id"]),
            "customer_message": msg,
            "true_intent": true_intent,
            "predicted_intent": pred_intent,
            "true_escalation": true_esc,
            "predicted_escalation": pred_esc,
            "top_similarity": round(top1_sims[-1], 3),
            "generated_reply": reply_text,
            "failure_type": "intent_mismatch" if pred_intent != true_intent else "escalation_miss"
        })

from sklearn.metrics import precision_recall_fscore_support
def esc_f1(y_true_bool, y_pred_str):
    y_pred = [1 if d == "HUMAN" else 0 for d in y_pred_str]
    y_true = [int(x) for x in y_true_bool]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

res = {
    "majority": evaluate_intent(golden["true_intent"], maj_pred),
    "tfidf": evaluate_intent(golden["true_intent"], tfidf_pred),
    "agent": evaluate_intent(golden["true_intent"], agent_intents),
    "escalation_agent": esc_f1(golden["true_escalation"], agent_escs),
    "retrieval_metrics": {
        "mean_top1_similarity": float(pd.Series(top1_sims).mean()),
        "mean_top5_similarity": float(pd.Series(top5_sims).mean()),
        "k": a.k
    },
    "meta": {
        "dataset": "thoughtvector/customer-support-on-twitter (TWCS)",
        "brand": "AmericanAir",
        "kb_size": len(conv),
        "golden_size": len(golden),
        "embedder": agent.emb_kind,
        "retriever": agent.retriever.backend,
        "llm_intent": agent.use_llm_intent
    },
}
print("\nBenchmark Results Summary:")
print(json.dumps(res, indent=2))

(ROOT / "results").mkdir(exist_ok=True)
save_confusion_matrix(golden["true_intent"], agent_intents, str(ROOT / "results" / "confusion_agent.png"))
save_confusion_matrix(golden["true_intent"], tfidf_pred, str(ROOT / "results" / "confusion_tfidf.png"))

# Save failure analysis
df_failures = pd.DataFrame(failures)
df_failures.to_csv(ROOT / "results" / "failures.csv", index=False)
print(f"Saved results/failures.csv ({len(df_failures)} failures out of {len(golden)} real queries)")

# LLM-as-a-judge evaluation
n = min(a.judge_n, len(golden))
scores = []
print(f"\nRunning LLM Judge on first {n} responses...")
for i in range(n):
    s = judge_response(golden.iloc[i]["customer_message"], cases_list[i], replies[i])
    scores.append({
        "id": int(golden.iloc[i]["id"]),
        "true_intent": golden.iloc[i]["true_intent"],
        "pred_intent": agent_intents[i],
        "escalation": agent_escs[i],
        "customer_message": golden.iloc[i]["customer_message"],
        "reply": replies[i],
        **s
    })
    time.sleep(1.2)
    if (i + 1) % 5 == 0 or (i + 1) == n:
        print(f"  Judged {i+1}/{n} responses...")

js = pd.DataFrame(scores)
js.to_csv(ROOT / "results" / "judge_scores.csv", index=False)
js.to_csv(ROOT / "results" / "llm_judge_results.csv", index=False)
res["judge_mean_overall"] = float(js["overall"].mean())
res["judge_means"] = {c: float(js[c].mean()) for c in ["correctness", "groundedness", "helpfulness", "tone", "completeness"]}
print("Judge dimension averages:", res["judge_means"], "overall:", round(res["judge_mean_overall"], 2))

# Human Validation / Agreement Check
# Per Hiver assignment guidelines: Never fabricate human agreement numbers.
# We supply a standardized human_review_template.csv for auditing.
human_scores_path = ROOT / "data" / "golden" / "human_scores.csv"
if human_scores_path.exists():
    try:
        df_human = pd.read_csv(human_scores_path)
        from evaluation.human_agreement import agreement
        sample_40 = js.head(len(df_human)).copy()
        m = df_human.merge(sample_40[["id", "overall"]].rename(columns={"overall": "llm_score"}), on="id")
        m = m.rename(columns={"human_overall": "human_score"})
        res["human_agreement"] = agreement(m)
        print("Human-to-Judge Agreement Metrics (on human_scores.csv):", res["human_agreement"])
    except Exception as e:
        print(f"Skipping human agreement calculation: {e}")
        res["human_agreement"] = "Human audit template prepared in data/golden/human_review_template.csv; multi-rater agreement designated as future work."
else:
    res["human_agreement"] = "Human audit template prepared in data/golden/human_review_template.csv; multi-rater agreement designated as future work."

with open(ROOT / "results" / "results.json", "w") as f:
    json.dump(res, f, indent=2)
print("Saved complete benchmark to results/results.json")

