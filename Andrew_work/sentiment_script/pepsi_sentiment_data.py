import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

reddit_sentiment_path = os.path.join(base_dir, "pepsi_sentiment_reddit.csv")
tweets_sentiment_path = os.path.join(base_dir, "pepsi_sentiment_tweets.csv")

reddit_sentiment_df = pd.read_csv(reddit_sentiment_path)
tweets_sentiment_df = pd.read_csv(tweets_sentiment_path)

reddit_sentiment_df['source'] = 'Reddit'
tweets_sentiment_df['source'] = 'Twitter'

columns_to_keep = [
    'text', 'date', 'sentiment_score_textblob', 'sentiment_score_vader',
    'sentiment_textblob', 'sentiment_vader', 'hour', 'weekday', 'source'
]
reddit_sentiment_df = reddit_sentiment_df[columns_to_keep]
tweets_sentiment_df = tweets_sentiment_df[columns_to_keep]

combined_sentiment_df = pd.concat([reddit_sentiment_df, tweets_sentiment_df], ignore_index=True)

reddit_post_counts = reddit_sentiment_df.groupby("date").size().rename("Reddit_Post_Count")
twitter_tweet_counts = tweets_sentiment_df.groupby("date").size().rename("Twitter_Tweet_Count")

daily_reddit_avg_sentiment = reddit_sentiment_df.groupby("date")["sentiment_score_vader"].mean().rename("Reddit_Avg_Sentiment")
daily_twitter_avg_sentiment = tweets_sentiment_df.groupby("date")["sentiment_score_vader"].mean().rename("Twitter_Avg_Sentiment")

keyword = "Pepsi"
reddit_keyword_counts = reddit_sentiment_df[reddit_sentiment_df["text"].str.contains(keyword, case=False, na=False)].groupby("date").size().rename("Reddit_Keyword_Count")
twitter_keyword_counts = tweets_sentiment_df[tweets_sentiment_df["text"].str.contains(keyword, case=False, na=False)].groupby("date").size().rename("Twitter_Keyword_Count")

additional_metrics = pd.concat([
    reddit_post_counts, twitter_tweet_counts,
    daily_reddit_avg_sentiment, daily_twitter_avg_sentiment,
    reddit_keyword_counts, twitter_keyword_counts
], axis=1).fillna(0)

combined_sentiment_df = combined_sentiment_df.merge(additional_metrics, on="date", how="left")

optimized_output_path = os.path.join(base_dir, "pepsi_sentiment_data.csv")
combined_sentiment_df.to_csv(optimized_output_path, index=False)

print(f"Optimized combined sentiment data saved to {optimized_output_path}")

print(combined_sentiment_df.info())
print(combined_sentiment_df.head())
