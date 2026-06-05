from datetime import datetime, timedelta, timezone

import pandas as pd


RAW_FILE = "DatasetIDF.xlsx"
OUTPUT_FILE = "data/processed/openagenda_events_clean.csv"


COLUMNS_TO_KEEP = [
    "Identifiant",
    "Slug",
    "URL canonique",
    "Titre",
    "Description",
    "Description longue",
    "Mots clés",
    "Première date - Début",
    "Première date - Fin",
    "Dernière date - Début",
    "Dernière date - Fin",
    "Nom du lieu",
    "Adresse",
    "Code postal",
    "Ville",
    "Département",
    "Région",
    "Pays",
    "État de l'événement",
    "Agenda d'origine (titre)",
    "Agenda d'origine (uid)",
    "Catégorie",
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Charge le fichier Excel OpenAgenda fourni.
    """
    return pd.read_excel(file_path)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve uniquement les colonnes utiles au système RAG.
    """
    available_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    return df[available_columns].copy()


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les dates et conserve uniquement les événements de moins d'un an.
    Les événements futurs sont conservés.
    """
    df["Première date - Début"] = pd.to_datetime(
        df["Première date - Début"],
        errors="coerce",
        utc=True
    )

    df["Dernière date - Fin"] = pd.to_datetime(
        df["Dernière date - Fin"],
        errors="coerce",
        utc=True
    )
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

    df = df[df["Première date - Début"].notna()]
    df = df[df["Première date - Début"] >= one_year_ago]

    return df


def clean_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les événements incomplets ou doublons.
    """
    df = df.dropna(subset=["Titre"])
    df = df.drop_duplicates(subset=["Identifiant"])

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
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def create_embedding_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un champ texte complet qui sera ensuite vectorisé.
    """
    df["text_for_embedding"] = (
        "Titre : " + df["Titre"] + "\n"
        "Description : " + df["Description"] + "\n"
        "Description longue : " + df["Description longue"] + "\n"
        "Mots clés : " + df["Mots clés"] + "\n"
        "Catégorie : " + df["Catégorie"] + "\n"
        "Lieu : " + df["Nom du lieu"] + ", " + df["Adresse"] + ", " + df["Ville"] + "\n"
        "Département : " + df["Département"] + "\n"
        "Région : " + df["Région"] + "\n"
        "Date de début : " + df["Première date - Début"].astype(str) + "\n"
        "Date de fin : " + df["Dernière date - Fin"].astype(str)
    )

    return df


def save_dataset(df: pd.DataFrame, output_file: str) -> None:
    """
    Sauvegarde le dataset nettoyé.
    """
    df.to_csv(output_file, index=False)


def main() -> None:
    df = load_dataset(RAW_FILE)

    print(f"Nombre initial d'événements : {len(df)}")

    df = select_columns(df)
    df = clean_dates(df)
    df = clean_content(df)
    df = create_embedding_text(df)

    save_dataset(df, OUTPUT_FILE)

    print(f"Nombre d'événements après nettoyage : {len(df)}")
    print(f"Dataset sauvegardé : {OUTPUT_FILE}")

    print(df[[
    "Titre",
    "Ville",
    "Première date - Début",
    "Catégorie"
]].head())


if __name__ == "__main__":
    main()