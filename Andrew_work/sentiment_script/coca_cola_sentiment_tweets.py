import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import seaborn as sns
import nltk
from pathlib import Path

nltk.download('vader_lexicon')

data_path = Path(__file__).resolve().parent.parent / "data" / "coca_cola_tweets.csv"
if not data_path.exists():
    raise FileNotFoundError(
        f"Source data not found: {data_path}. "
        "Raw social-media files are excluded from the public portfolio; see README.md."
    )

df = pd.read_csv(data_path)
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at'].dt.date

df['text'] = df['text'].fillna('').astype(str)

def analyze_sentiments(data):
    sia = SentimentIntensityAnalyzer()

    data['sentiment_score_textblob'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    data['sentiment_score_vader'] = data['text'].apply(lambda x: sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0)

    data['sentiment_textblob'] = data['sentiment_score_textblob'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
    )
    data['sentiment_vader'] = data['sentiment_score_vader'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
    )
    return data

df = analyze_sentiments(df)

keyword = input("Enter a keyword to filter tweets (e.g., 'Coke'): ").strip()
filtered_df = df[df['text'].str.contains(keyword, case=False, na=False)]
print(f"Filtered {len(filtered_df)} tweets containing the keyword: {keyword}")

def create_visualizations(data, prefix="coca_cola_tweets"):
    daily_counts = data['date'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    daily_counts.plot(kind='bar', color='skyblue', title='Daily Tweet Counts')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for sentiment_type, column in [('TextBlob', 'sentiment_score_textblob'), ('VADER', 'sentiment_score_vader')]:
        daily_sentiment = data.groupby('date')[column].mean()
        plt.figure(figsize=(10, 5))
        daily_sentiment.plot(kind='line', marker='o', title=f'Daily Sentiment Trend ({sentiment_type})', color='green')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    sentiment_distribution = data['sentiment_vader'].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_distribution.plot(kind='bar', color=['red', 'blue', 'green'], title='Sentiment Distribution (VADER)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    data['hour'] = data['created_at'].dt.hour
    hourly_tweets = data['hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    hourly_tweets.plot(kind='bar', color='orange', title='Hourly Tweet Distribution')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    combined_sentiment = data.groupby('date')[['sentiment_score_textblob', 'sentiment_score_vader']].mean()
    plt.figure(figsize=(10, 5))
    combined_sentiment.plot(kind='line', marker='o', title='Combined Sentiment Trends', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    from collections import Counter
    keyword_words = ' '.join(filtered_df['text']).lower().split()
    common_words = Counter(keyword_words).most_common(10)
    words, counts = zip(*common_words)
    plt.figure(figsize=(8, 5))
    plt.bar(words, counts, color='purple')
    plt.title(f"Top 10 Words in Tweets Containing '{keyword}'")
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    hourly_sentiment = data.groupby('hour')[['sentiment_score_textblob', 'sentiment_score_vader']].mean()
    plt.figure(figsize=(10, 5))
    hourly_sentiment.plot(kind='line', marker='o', title='Hourly Sentiment Comparison', linewidth=2)
    plt.xlabel('Hour')
    plt.ylabel('Sentiment Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    data['weekday'] = data['created_at'].dt.day_name()
    pivot_table = data.pivot_table(index='hour', columns='weekday', values='sentiment_score_vader', aggfunc='mean')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap of VADER Sentiments by Hour and Day')
    plt.xlabel('Weekday')
    plt.ylabel('Hour')

create_visualizations(df)

output_path = Path(__file__).resolve().parent / "coca_cola_sentiment_tweets.csv"
df.to_csv(output_path, index=False)
print("Final analyzed data saved to: coca_cola_sentiment_tweets.csv")
