import pandas as pd

DATASET_PATH = "data/processed/openagenda_events_clean.csv"


def test_events_less_than_one_year_old():
    """
    Vérifie que tous les événements du dataset
    sont récents (moins d'un an).

    Une tolérance d'un jour est appliquée afin
    d'éviter les problèmes liés aux fuseaux horaires
    et aux heures exactes d'exécution.
    """

    df = pd.read_csv(DATASET_PATH)

    df["Première date - Début"] = pd.to_datetime(
        df["Première date - Début"],
        utc=True,
    )

    one_year_ago = (
        pd.Timestamp.now(tz="UTC").normalize()
        - pd.Timedelta(days=366)
    )

    old_events = df[
        df["Première date - Début"] < one_year_ago
    ]

    assert len(old_events) == 0


def test_events_are_in_idf():
    """
    Vérifie que tous les événements appartiennent
    à la région Île-de-France.
    """

    df = pd.read_csv(DATASET_PATH)

    assert (
        df["Région"]
        .fillna("")
        .eq("Île-de-France")
        .all()
    )