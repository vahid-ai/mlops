"""Backfill features using Spark/DataFusion jobs."""

def backfill(start_date: str, end_date: str) -> None:
    print(f"Backfilling features from {start_date} to {end_date}")
