"""Central config. All thresholds in one place so evaluation can ablate them."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_GOLDEN = ROOT / "data" / "golden"
RESULTS = ROOT / "results"

BRAND_NAME = os.getenv("BRAND_NAME", "AmericanAir")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # used only if sentence-transformers installed
TOP_K = 5
INTENT_CONFIDENCE_THRESHOLD = 0.55
RETRIEVAL_SIMILARITY_THRESHOLD = 0.15  # cosine on TF-IDF; tuned on sample (see decision_log #6)
RANDOM_STATE = 42
