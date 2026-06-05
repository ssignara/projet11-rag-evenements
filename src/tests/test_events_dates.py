from datetime import datetime, timedelta, timezone
import pandas as pd


def test_events_less_than_one_year_old():
    df = pd.read_csv("data/processed/openagenda_events_clean.csv")

    df["Première date - Début"] = pd.to_datetime(
        df["Première date - Début"],
        utc=True
    )

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

    old_events = df[df["Première date - Début"] < one_year_ago]

    assert len(old_events) == 0