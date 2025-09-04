import schedule
import time
from pipeline import run_pipeline

# Schedule the pipeline to run once every day at 2:00 AM
schedule.every().day.at("02:00").do(run_pipeline)

print("🕒 Scheduler started. The pipeline will run daily at 2:00 AM.")

while True:
    schedule.run_pending()
    time.sleep(60) # check every minute for a pending job