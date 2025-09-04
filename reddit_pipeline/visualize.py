import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# -----------------------------
# Step 1: Load Processed Data from MongoDB
# -----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "social_media_db"
COLLECTION_NAME = "reddit_daily_sentiment"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

print("📊 Fetching and preparing data for visualization...")

data = list(collection.find())
if not data:
    print("❌ No data found in MongoDB collection. Run the pipeline first!")
    exit()

df = pd.DataFrame(data)

# Ensure date column is datetime
if "date" not in df.columns:
    print("❌ 'date' column missing in MongoDB data!")
    exit()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Aggregate data by date
daily_summary = df.groupby("date").agg(
    post_count=("post_count", "sum"),
    avg_sentiment_score=("avg_sentiment_score", "mean")
).sort_index()

print("✅ Data prepared successfully.")

# -----------------------------
# Step 2: Generate Visualizations (Corrected Section)
# -----------------------------
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True) # Increased height a bit for readability

# --- Plot 1: Daily Post Count (Bar Chart) ---
# CHANGE: Use ax1.bar() instead of sns.barplot() to correctly handle the DatetimeIndex.
ax1.bar(
    daily_summary.index, 
    daily_summary["post_count"], 
    color="skyblue",
    width=0.8 # Adjust width for better appearance if needed
)
ax1.set_title("Daily Post Count", fontsize=16)
ax1.set_ylabel("Number of Posts")
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks for top plot


# --- Plot 2: Daily Average Sentiment Score (Line Chart) ---
# CHANGE: Plot directly against the DatetimeIndex. The manual conversion to numbers is no longer needed.
ax2.plot(
    daily_summary.index, 
    daily_summary["avg_sentiment_score"], 
    marker="o", 
    linestyle="-", 
    color="coral", 
    label="Avg Sentiment Score"
)
ax2.axhline(0, color="grey", linestyle="--", linewidth=1, label="Neutral Sentiment")

# CHANGE: The `ax2.xaxis_date()` call is now removed as it's redundant.
# Matplotlib automatically recognizes the datetime objects.

# Format the shared x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax2.xaxis.set_major_locator(mdates.AutoDateLocator()) # Improve tick placement automatically
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

ax2.set_title("Daily Average Sentiment Score", fontsize=16)
ax2.set_xlabel("Date")
ax2.set_ylabel("Average Sentiment Score")
ax2.legend()


# -----------------------------
# Step 3: Show Plot
# -----------------------------
plt.tight_layout(pad=2.0) # Add some padding
plt.show()

# Close MongoDB connection
client.close()