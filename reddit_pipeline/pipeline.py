
import praw
import pandas as pd
import os
import re
import nltk
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient, UpdateOne
from prawcore.exceptions import NotFound

# --- CONFIGURATION ---
load_dotenv()
CONFIG = {
    "SUBREDDIT_NAME": "technology",
    "SEARCH_KEYWORD": "AI",
    "POST_LIMIT": 200,
    "MONGO_URI": os.getenv("MONGO_URI"),
    "DB_NAME": "social_media_db",
    "COLLECTION_NAME": "reddit_daily_sentiment"
}

# --- SETUP ---
# nltk.download('stopwords')
# nltk.download('wordnet')

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- EXTRACTION ---
def extract_data(subreddit_name, search_keyword, limit):
    """Extracts post data from a specified subreddit with error handling."""
    posts_data = []
    print(f"🔍 Extracting data from r/{subreddit_name} for keyword: '{search_keyword}'...")
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.search(search_keyword, limit=limit):
            posts_data.append({
                'post_id': post.id,
                'title': post.title,
                'selftext': post.selftext,
                'created_utc': post.created_utc,
                'score': post.score,
                'url': post.url
            })
        df = pd.DataFrame(posts_data)
        print(f"✅ Extracted {len(df)} posts.")
        return df
    except NotFound:
        print(f"❌ Error: Subreddit r/{subreddit_name} not found.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"❌ An unexpected error occurred during extraction: {e}")
        return pd.DataFrame()

# --- TRANSFORMATION ---
def clean_text(text):
    """Cleans text data for analysis."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def analyze_sentiment(text):
    """Analyzes sentiment and returns the compound score."""
    return analyzer.polarity_scores(text)['compound']

def transform_data(df):
    """Applies cleaning and sentiment analysis to the DataFrame."""
    print("✨ Transforming data...")
    df['full_text'] = df['title'] + ' ' + df['selftext']
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    df['sentiment_score'] = df['cleaned_text'].apply(analyze_sentiment)
    # Convert to datetime object (without .dt.date to keep time info if needed)
    df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
    print("✅ Transformation complete.")
    return df

# --- AGGREGATION ---
def aggregate_data(df):
    """Aggregates data to get daily metrics."""
    print("📊 Aggregating data...")
    # Use pd.Grouper to group by calendar day from datetime objects
    daily_summary = df.groupby(pd.Grouper(key='created_date', freq='D')).agg(
        post_count=('post_id', 'count'),
        avg_sentiment_score=('sentiment_score', 'mean')
    ).reset_index()
    daily_summary = daily_summary.rename(columns={'created_date': 'date'})
    print("✅ Aggregation complete.")
    return daily_summary

# --- LOADING ---
def load_to_mongodb(data, db_name, collection_name, mongo_uri):
    """Loads a DataFrame into a MongoDB collection using efficient bulk operations."""
    if data.empty:
        print("ℹ️ No data to load. Skipping MongoDB operation.")
        return
        
    print("📦 Loading data into MongoDB...")
    client = None # Initialize client to None for the finally block
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Prepare a list of bulk write operations
        operations = []
        for record in data.to_dict("records"):
            # The filter finds the document to update based on its date
            # The record itself is used in the $set operator to update the data
            op = UpdateOne(
                {"date": record['date']},
                {"$set": record},
                upsert=True
            )
            operations.append(op)
        
        # Execute all operations in a single request
        result = collection.bulk_write(operations)
        print(f"✅ Bulk write complete. Matched: {result.matched_count}, Upserted: {result.upserted_count}")

    except Exception as e:
        print(f"❌ Error loading data to MongoDB: {e}")
    finally:
        if client:
            client.close()

# --- MAIN EXECUTION ---
def run_pipeline():
    """Executes the full ETL pipeline using parameters from the CONFIG dictionary."""
    print("🚀 Starting the social media data pipeline...")
    raw_df = extract_data(
        subreddit_name=CONFIG["SUBREDDIT_NAME"],
        search_keyword=CONFIG["SEARCH_KEYWORD"],
        limit=CONFIG["POST_LIMIT"]
    )
    if not raw_df.empty:
        transformed_df = transform_data(raw_df)
        aggregated_df = aggregate_data(transformed_df)
        load_to_mongodb(
            aggregated_df,
            db_name=CONFIG["DB_NAME"],
            collection_name=CONFIG["COLLECTION_NAME"],
            mongo_uri=CONFIG["MONGO_URI"]
        )
        print("🎉 Pipeline run completed successfully!")

if __name__ == '__main__':
    run_pipeline()