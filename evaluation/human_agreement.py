"""Human vs LLM agreement: Pearson r + quadratic-weighted kappa on overall 1-5."""
import pandas as pd


def agreement(df: pd.DataFrame):
    from sklearn.metrics import cohen_kappa_score
    corr = float(df[["human_score", "llm_score"]].corr().iloc[0, 1])
    try:
        kappa = float(cohen_kappa_score(df["human_score"], df["llm_score"], weights="quadratic"))
    except Exception:
        kappa = float("nan")
    return {"pearson": corr, "quadratic_kappa": kappa, "n": len(df),
            "mean_human": float(df["human_score"].mean()),
            "mean_llm": float(df["llm_score"].mean())}
