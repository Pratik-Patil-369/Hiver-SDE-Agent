# Raw data (not committed)
Place `twcs.csv` here from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
Expected columns: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
Then run: `python scripts/build_conversations.py --brand AmericanAir`
The repo ships an offline sample in data/processed/conversations.csv so everything reproduces without this file.
