"""Build data/processed/conversations.csv from data/raw/twcs.csv (real data path).

Usage: python scripts/build_conversations.py [--brand AmericanAir]
Falls back to shipped sample if raw file missing.
"""
import argparse
from pathlib import Path
import pandas as pd

from src.data_loader import load_dataset, show_basic_statistics
from src.preprocessing import build_conversations_from_twcs

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--brand", default="AmericanAir")
p.add_argument("--raw", default="data/raw/twcs.csv")
p.add_argument("--out", default="data/processed/conversations.csv")
a = p.parse_args()

raw = ROOT / a.raw
out = ROOT / a.out
if not raw.exists():
    print(f"Raw file not found: {raw}")
    print("Using shipped sample at data/processed/conversations.csv (offline mode).")
    print("To use real data, run: python scripts/download_data.py")
    raise SystemExit(0)

df = load_dataset(raw)
show_basic_statistics(df)
conv = build_conversations_from_twcs(df, a.brand)
print(f"Brand={a.brand} conversations={len(conv)}")
out.parent.mkdir(parents=True, exist_ok=True)
conv.to_csv(out, index=False)
print(f"Saved -> {out}")
print(conv.head(3).to_string())
