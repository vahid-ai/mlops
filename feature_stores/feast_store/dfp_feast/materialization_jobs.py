"""Materialization jobs for Feast online store."""

def materialize_to_online(start_date: str, end_date: str) -> None:
    print(f"Materializing from {start_date} to {end_date}")
