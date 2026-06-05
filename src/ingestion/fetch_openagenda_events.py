import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")

# À remplacer par l'UID de l'agenda OpenAgenda choisi
AGENDA_UID = "12345678"

BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"


def fetch_events(size: int = 100) -> list:
    """
    Récupère les événements OpenAgenda récents et à venir.
    """
    if not API_KEY:
        raise ValueError("La clé OPENAGENDA_API_KEY est absente du fichier .env")

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

    params = {
        "key": API_KEY,
        "size": size,
        "timings[gte]": one_year_ago.isoformat(),
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("events", [])


def normalize_events(events: list) -> pd.DataFrame:
    """
    Transforme les événements bruts OpenAgenda en DataFrame structuré.
    """
    rows = []

    for event in events:
        title = event.get("title", {}).get("fr", "")
        description = event.get("description", {}).get("fr", "")
        long_description = event.get("longDescription", {}).get("fr", "")

        location = event.get("location", {}) or {}
        timings = event.get("timings", [])
        first_timing = timings[0] if timings else {}

        rows.append(
            {
                "uid": event.get("uid"),
                "title": title,
                "description": description,
                "long_description": long_description,
                "location_name": location.get("name"),
                "city": location.get("city"),
                "postal_code": location.get("postalCode"),
                "department": location.get("department"),
                "region": location.get("region"),
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
                "begin_date": first_timing.get("begin"),
                "end_date": first_timing.get("end"),
                "url": event.get("canonicalUrl"),
            }
        )

    return pd.DataFrame(rows)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les événements et ne conserve que les événements de moins d'un an.
    Les événements futurs sont conservés.
    """
    if df.empty:
        return df

    df = df.copy()

    df = df.dropna(subset=["title", "begin_date"])
    df = df.drop_duplicates(subset=["uid"])

    df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce", utc=True)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

    df = df[df["begin_date"] >= one_year_ago]

    df["text_for_embedding"] = (
        "Titre : " + df["title"].fillna("") + "\n"
        "Description : " + df["description"].fillna("") + "\n"
        "Description longue : " + df["long_description"].fillna("") + "\n"
        "Lieu : " + df["location_name"].fillna("") + ", " + df["city"].fillna("") + "\n"
        "Date : " + df["begin_date"].astype(str)
    )

    return df


def save_events(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Sauvegarde les fichiers raw et processed.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df_raw.to_csv("data/raw/openagenda_events_raw.csv", index=False)
    df_clean.to_csv("data/processed/openagenda_events_clean.csv", index=False)


if __name__ == "__main__":
    events = fetch_events(size=100)

    df_raw = normalize_events(events)
    df_clean = clean_events(df_raw)

    save_events(df_raw, df_clean)

    print(f"{len(df_raw)} événements récupérés")
    print(f"{len(df_clean)} événements propres sauvegardés")
    print("Fichiers générés :")
    print("- data/raw/openagenda_events_raw.csv")
    print("- data/processed/openagenda_events_clean.csv")