"""Full evaluation: baselines vs agent on golden set + escalation + LLM judge + agreement.

Usage: python scripts/run_evaluation.py [--judge-n 200] [--k 5]
Outputs: results/results.json, results/confusion_*.png, results/judge_scores.csv
"""
import argparse, json, sys
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
# Agent
agent = Agent(conv, use_llm_intent=False)
agent_intents, agent_escs, replies, cases_list = [], [], [], []
for msg in golden["customer_message"].tolist():
    r = agent.handle(msg, k=a.k)
    agent_intents.append(r["intent"]["intent"])
    agent_escs.append(r["escalation"]["decision"])
    replies.append(r["reply"]["reply"])
    ev = "\n".join(c["record"].get("agent_response", "") for c in r["historical_cases"][:3])
    cases_list.append(ev)

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
    "meta": {"kb": len(conv), "golden": len(golden), "embedder": agent.emb_kind,
             "retriever": agent.retriever.backend, "llm_intent": agent.use_llm_intent},
}
print(json.dumps(res, indent=2))

(ROOT / "results").mkdir(exist_ok=True)
save_confusion_matrix(golden["true_intent"], agent_intents, str(ROOT / "results" / "confusion_agent.png"))
save_confusion_matrix(golden["true_intent"], tfidf_pred, str(ROOT / "results" / "confusion_tfidf.png"))

# LLM judge on first judge-n replies
n = min(a.judge_n, len(golden))
scores = []
for i in range(n):
    s = judge_response(golden.iloc[i]["customer_message"], cases_list[i], replies[i])
    scores.append({"id": int(golden.iloc[i]["id"]), "true_intent": golden.iloc[i]["true_intent"],
                   "pred_intent": agent_intents[i], "escalation": agent_escs[i],
                   "reply": replies[i], **s})
js = pd.DataFrame(scores)
js.to_csv(ROOT / "results" / "judge_scores.csv", index=False)
res["judge_mean_overall"] = float(js["overall"].mean())
res["judge_means"] = {c: float(js[c].mean()) for c in ["correctness", "groundedness", "helpfulness", "tone", "completeness"]}
print("Judge means:", res["judge_means"], "overall:", res["judge_mean_overall"])

# Human agreement if human scores present
hp = ROOT / "data" / "golden" / "human_scores.csv"
if hp.exists():
    from evaluation.human_agreement import agreement
    h = pd.read_csv(hp)
    m = h.merge(js[["id", "overall"]].rename(columns={"overall": "llm_score"}),
                left_on="id", right_on="id").rename(columns={"human_overall": "human_score"})
    res["human_agreement"] = agreement(m)
    print("Human agreement:", res["human_agreement"])
else:
    print("No human_scores.csv — creating template for 40-item human review.")
    js.sample(min(40, len(js)), random_state=42)[["id"]].to_csv(
        ROOT / "data" / "golden" / "human_review_template.csv", index=False)
    # copy replies for reviewers
    js.sample(min(40, len(js)), random_state=42).to_csv(
        ROOT / "data" / "golden" / "human_review_items.csv", index=False)

with open(ROOT / "results" / "results.json", "w") as f:
    json.dump(res, f, indent=2)
print("Saved results/results.json")
