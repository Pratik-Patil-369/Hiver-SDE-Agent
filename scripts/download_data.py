"""Download helper. Tries Kaggle API; else prints manual steps.

Expected raw file: data/raw/twcs.csv (Customer Support on Twitter,
thoughtvector/customer-support-on-twitter, columns: tweet_id,author_id,
inbound,created_at,text,response_tweet_id,in_response_to_tweet_id)
"""
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

print(f"Raw dir: {RAW}")
print("1) Manual (recommended): https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
print("   Download twcs.csv -> save as data/raw/twcs.csv")
print("2) CLI: kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip")
try:
    import kaggle  # type: ignore
    print("kaggle package found; attempting download...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files("thoughtvector/customer-support-on-twitter",
                               path=str(RAW), unzip=True)
    print("Downloaded.")
except Exception as e:
    print(f"Auto-download skipped ({e}). Follow manual steps above.")
    print("The repo runs fully offline on the shipped sample in data/processed/conversations.csv.")
