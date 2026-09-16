"""Minimal Streamlit demo (P1). Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.agent import Agent

st.set_page_config(page_title="Hiver AI Support Agent (AmericanAir)", layout="wide")
st.title("AI Customer Support Agent — AmericanAir")

@st.cache_resource
def load_agent():
    conv = pd.read_csv(ROOT / "data" / "processed" / "conversations.csv")
    return Agent(conv, use_llm_intent=True)

agent = load_agent()
msg = st.text_area("Customer message", placeholder="e.g. My flight AA123 was cancelled, what now?")
if st.button("Analyze"):
    if not msg.strip():
        st.warning("Enter a message.")
    else:
        r = agent.handle(msg)
        st.subheader("Intent")
        st.write(f"{r['intent']['intent']} (conf {r['intent']['confidence']:.2f})")
        st.subheader("Draft reply")
        st.info(r["reply"]["reply"])
        st.caption(r["reply"]["evidence_summary"])
        st.subheader("Decision")
        if r["escalation"]["decision"] == "HUMAN":
            st.error("HUMAN — " + r["escalation"]["reason"])
        else:
            st.success("AUTO-HANDLE — " + r["escalation"]["reason"])
        with st.expander("Historical evidence"):
            for c in r["historical_cases"]:
                st.write(f"{c['score']:.2f} — {c['record'].get('customer_message','')[:160]}")
