"""Preprocessing + conversation building.

Real twcs.csv schema (thoughtvector/customer-support-on-twitter):
  tweet_id, author_id, inbound, created_at, text,
  response_tweet_id, in_response_to_tweet_id

- inbound=True  -> customer message
- inbound=False -> brand/support message
- author_id     -> e.g. 'AmericanAir', 'sprintcare', or customer handle
- in_response_to_tweet_id / response_tweet_id link threads.

build_conversations() works on the REAL schema when present, and on our
shipped sample (data/processed/conversations.csv) which already uses the
output schema below.
Output schema: conversation_id, customer_message, agent_response, timestamp
"""
import pandas as pd


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s


def build_conversations_from_twcs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    # Normalize column names defensively
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_text = col("text")
    c_author = col("author_id", "author")
    c_inbound = col("inbound")
    c_created = col("created_at", "timestamp", "date")
    c_tweet = col("tweet_id", "id")
    c_inresp = col("in_response_to_tweet_id", "in_response_to", "response_to")
    c_resp = col("response_tweet_id", "response_id")

    assert c_text and c_author, f"Unexpected columns: {df.columns.tolist()}"

    df = df.copy()
    df["_text"] = df[c_text].map(clean_text)
    df = df[df["_text"] != ""].drop_duplicates(subset=["_text", c_author] if c_tweet is None else None)

    # Restrict to threads involving the brand
    brand_mask = df[c_author].astype(str).str.lower() == brand.lower()
    # Keep customer tweets directed at brand + brand replies. Heuristic: keep rows
    # where author is brand OR (inbound customer tweet that got/was a response).
    if c_inbound is not None:
        inbound = df[c_inbound].astype(str).str.lower().isin(["true", "1", "t"])
    else:
        inbound = ~brand_mask

    # Pair each customer message with the brand's direct response if linkable,
    # else with the next brand tweet in the same thread / time window.
    df["_inbound"] = inbound
    df["_brand"] = brand_mask
    if c_created is not None:
        df["_ts"] = pd.to_datetime(df[c_created], errors="coerce", utc=True)

    rows = []
    if c_inresp is not None and c_tweet is not None:
        by_id = df.set_index(c_tweet)
        for _, r in df[~df["_inbound"] | (~df["_brand"])].iterrows():
            pass  # placeholder for clarity; real pairing below
        # Customer rows that mention/are linked to brand
        cust = df[(df["_inbound"]) & (~df["_brand"])]
        for _, cr in cust.iterrows():
            agent_resp = ""
            ts = cr["_ts"] if "_ts" in df.columns else ""
            # 1) direct response id pointer
            if c_resp is not None and pd.notna(cr.get(c_resp, None)):
                rid = cr[c_resp]
                if rid in by_id.index:
                    agent_resp = by_id.loc[rid, "_text"]
            # 2) brand tweet whose in_response_to points at this tweet
            if not agent_resp and c_tweet is not None:
                match = df[(df["_brand"]) & (df[c_inresp].astype(str) == str(cr[c_tweet]))]
                if len(match):
                    agent_resp = match.iloc[0]["_text"]
            if agent_resp:
                rows.append({
                    "conversation_id": str(cr[c_tweet]) if c_tweet else f"c{len(rows)}",
                    "customer_message": cr["_text"],
                    "agent_response": agent_resp,
                    "timestamp": str(ts) if pd.notna(ts) else "",
                })
    else:
        # Fallback: chronological pairing within brand-involved slice
        cust = df[(df["_inbound"]) & (~df["_brand"])].copy()
        supp = df[(df["_brand"])].copy()
        if "_ts" in df.columns:
            cust = cust.sort_values("_ts"); supp = supp.sort_values("_ts")
        for i, (_, cr) in enumerate(cust.iterrows()):
            ar = supp.iloc[i]["_text"] if i < len(supp) else ""
            if ar:
                rows.append({
                    "conversation_id": f"c{i}",
                    "customer_message": cr["_text"],
                    "agent_response": ar,
                    "timestamp": str(cr["_ts"]) if "_ts" in df.columns else "",
                })

    conv = pd.DataFrame(rows, columns=["conversation_id", "customer_message", "agent_response", "timestamp"])
    conv = conv[conv["customer_message"].str.len() > 0]
    conv = conv.drop_duplicates(subset=["customer_message", "agent_response"])
    return conv
