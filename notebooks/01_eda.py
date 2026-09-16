"""01_eda.py — run on real twcs.csv when available; prints brand stats for brand selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.data_loader import load_dataset, show_basic_statistics

raw = Path("data/raw/twcs.csv")
if not raw.exists():
    print("data/raw/twcs.csv missing — offline sample stats instead:")
    conv = pd.read_csv("data/processed/conversations.csv")
    print(conv.shape, conv.head(2).to_string())
else:
    df = load_dataset(raw)
    show_basic_statistics(df)
    print(df["author_id"].value_counts().head(20))
