import os
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")
AGENDA_UID = os.getenv("OPENAGENDA_AGENDA_UID", "56500817")

BASE_URL = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

RAW_OUTPUT_FILE = "data/raw/openagenda_events_raw.csv"
CLEAN_OUTPUT_FILE = "data/processed/openagenda_events_clean.csv"


def fetch_events_from_api(size: int = 100, max_pages: int = 30) -> list:
    if not API_KEY:
        raise ValueError("OPENAGENDA_API_KEY est absente du fichier .env")

    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

    all_events = []
    offset = 0

    for page in range(max_pages):
        params = {
            "key": API_KEY,
            "size": size,
            "offset": offset,
            "timings[gte]": one_year_ago.isoformat(),
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        events = response.json().get("events", [])

        if not events:
            break

        all_events.extend(events)
        print(f"Page {page + 1} : {len(events)} événements récupérés")

        offset += size

    return all_events


def get_text_value(value) -> str:
    if isinstance(value, dict):
        return value.get("fr") or value.get("en") or ""
    if isinstance(value, str):
        return value
    return ""


def normalize_region(value: str) -> str:
    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = value.replace("-", " ")
    value = " ".join(value.split())
    return value


def normalize_events(events: list) -> pd.DataFrame:
    rows = []

    for event in events:
        location = event.get("location") or {}

        first_timing = event.get("firstTiming") or {}
        last_timing = event.get("lastTiming") or {}

        keywords = event.get("keywords") or []

        if isinstance(keywords, dict):
            keywords = keywords.get("fr") or keywords.get("en") or []

        if isinstance(keywords, list):
            keywords = ", ".join([str(keyword) for keyword in keywords])
        else:
            keywords = str(keywords) if keywords else ""

        rows.append(
            {
                "Identifiant": event.get("uid"),
                "Slug": event.get("slug"),
                "URL canonique": event.get("canonicalUrl"),
                "Titre": get_text_value(event.get("title")),
                "Description": get_text_value(event.get("description")),
                "Description longue": get_text_value(event.get("longDescription")),
                "Mots clés": keywords,
                "Première date - Début": first_timing.get("begin"),
                "Première date - Fin": first_timing.get("end"),
                "Dernière date - Début": last_timing.get("begin"),
                "Dernière date - Fin": last_timing.get("end"),
                "Nom du lieu": location.get("name"),
                "Adresse": location.get("address"),
                "Code postal": location.get("postalCode"),
                "Ville": location.get("city"),
                "Département": location.get("department"),
                "Région": "Île-de-France",
                "Pays": location.get("countryCode"),
                "État de l'événement": event.get("status"),
                "Agenda d'origine (uid)": event.get("originAgenda", {}).get("uid")
                if isinstance(event.get("originAgenda"), dict)
                else AGENDA_UID,
                "Agenda d'origine (titre)": event.get("originAgenda", {}).get("title")
                if isinstance(event.get("originAgenda"), dict)
                else "",
                "Catégorie": get_text_value(event.get("category")),
                "Thématique": event.get("thematique"),
            }
        )

    return pd.DataFrame(rows)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["Première date - Début"] = pd.to_datetime(
        df["Première date - Début"],
        errors="coerce",
        utc=True,
    )

    df["Dernière date - Fin"] = pd.to_datetime(
        df["Dernière date - Fin"],
        errors="coerce",
        utc=True,
    )

    one_year_ago = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=365)

    df = df.dropna(subset=["Identifiant", "Titre", "Première date - Début"])
    df = df.drop_duplicates(subset=["Identifiant"])

    df = df[df["Première date - Début"] >= one_year_ago]

    df["region_normalized"] = df["Région"].fillna("").apply(normalize_region)

    allowed_regions = {
        "ile de france",
        "idf",
    }

    df = df[df["region_normalized"].isin(allowed_regions)]

    text_columns = [
        "Titre",
        "Description",
        "Description longue",
        "Mots clés",
        "Nom du lieu",
        "Adresse",
        "Ville",
        "Département",
        "Région",
        "Catégorie",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    df["text_for_embedding"] = (
        "Titre : " + df["Titre"] + "\n"
        "Description : " + df["Description"] + "\n"
        "Description longue : " + df["Description longue"] + "\n"
        "Mots clés : " + df["Mots clés"] + "\n"
        "Catégorie : " + df["Catégorie"] + "\n"
        "Lieu : " + df["Nom du lieu"] + ", " + df["Adresse"] + ", " + df["Ville"] + "\n"
        "Département : " + df["Département"].fillna("").astype(str) + "\n"
        "Région : " + df["Région"] + "\n"
        "Date de début : " + df["Première date - Début"].astype(str) + "\n"
        "Date de fin : " + df["Dernière date - Fin"].astype(str)
    )

    return df


def save_datasets(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df_raw.to_csv(RAW_OUTPUT_FILE, index=False)
    df_clean.to_csv(CLEAN_OUTPUT_FILE, index=False)


def main() -> None:
    events = fetch_events_from_api(size=100, max_pages=300)

    df_raw = normalize_events(events)
    df_clean = clean_events(df_raw)

    save_datasets(df_raw, df_clean)

    print(f"Nombre d'événements récupérés via API : {len(df_raw)}")
    print(f"Nombre d'événements après nettoyage : {len(df_clean)}")
    print(f"Dataset brut sauvegardé : {RAW_OUTPUT_FILE}")
    print(f"Dataset propre sauvegardé : {CLEAN_OUTPUT_FILE}")

    if not df_clean.empty:
        print(df_clean[["Titre", "Ville", "Première date - Début", "Région"]].head())


if __name__ == "__main__":
    main()