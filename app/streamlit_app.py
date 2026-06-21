import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

load_dotenv()

DATASET_PATH = "data/processed/openagenda_events_clean.csv"
VECTORSTORE_PATH = "vectorstore/faiss_index"


@st.cache_data
def load_events():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_vectorstore():
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def haversine_distance(lat1, lon1, lat2, lon2):
    radius_earth = 6371

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return radius_earth * c


def geocode_address(address):
    url = "https://api-adresse.data.gouv.fr/search/"

    params = {
        "q": address,
        "limit": 1,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not data.get("features"):
        return None

    coordinates = data["features"][0]["geometry"]["coordinates"]
    lon, lat = coordinates

    return lat, lon


def get_city_coordinates(df, city):
    city_events = df[
        df["Ville"].fillna("").str.lower() == city.lower()
    ]

    city_events = city_events.dropna(subset=["Latitude", "Longitude"])

    if city_events.empty:
        return None

    lat = city_events["Latitude"].mean()
    lon = city_events["Longitude"].mean()

    return lat, lon


def build_context(query, user_lat=None, user_lon=None, radius_km=None, k=80):
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(query, k=k)

    filtered_docs = []

    for doc in docs:
        metadata = doc.metadata

        event_lat = metadata.get("latitude")
        event_lon = metadata.get("longitude")

        if user_lat is not None and user_lon is not None and radius_km is not None:
            try:
                distance = haversine_distance(
                    float(user_lat),
                    float(user_lon),
                    float(event_lat),
                    float(event_lon),
                )

                if distance <= radius_km:
                    metadata["distance_km"] = round(distance, 2)
                    filtered_docs.append(doc)

            except (TypeError, ValueError):
                continue
        else:
            filtered_docs.append(doc)

    filtered_docs = filtered_docs[:5]

    context = "\n\n".join(
        doc.page_content
        + f"\nDistance utilisateur : {doc.metadata.get('distance_km', 'non calculée')} km"
        for doc in filtered_docs
    )

    return context, filtered_docs


def generate_answer(query, context):
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
Tu es un assistant spécialisé dans la recommandation d'événements culturels.

Date du jour : {today}

Tu dois répondre uniquement à partir des événements fournis dans le contexte.
Ne propose jamais un événement qui n'apparaît pas dans le contexte.

Si une distance est disponible, mentionne-la dans la recommandation.
Si aucun événement pertinent n'est disponible, indique clairement :
"Aucun événement pertinent n'a été trouvé dans les données disponibles."

Contexte :
{context}

Question :
{query}

Réponse structurée :
"""

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
    )

    response = llm.invoke(prompt)

    return response.content


st.set_page_config(
    page_title="Assistant Culturel RAG",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 Assistant de recommandation d'événements culturels")

st.write(
    "Posez une question en langage naturel et affinez la recherche par ville, adresse ou distance."
)

df = load_events()

villes = sorted(
    df["Ville"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

st.sidebar.header("📍 Localisation")

mode_localisation = st.sidebar.radio(
    "Mode de localisation",
    [
        "Toute l'Île-de-France",
        "Choisir une ville",
        "Saisir une adresse complète",
    ],
)

selected_city = None
address = None
user_lat = None
user_lon = None

if mode_localisation == "Choisir une ville":
    selected_city = st.sidebar.selectbox(
        "Ville",
        villes,
    )

elif mode_localisation == "Saisir une adresse complète":
    address = st.sidebar.text_input(
        "Adresse complète",
        placeholder="Exemple : 10 rue de Rivoli, 75004 Paris",
    )

radius_km = st.sidebar.selectbox(
    "Rayon autour du lieu choisi",
    [5, 10, 20, 50, 100],
    index=2,
)

st.sidebar.header("🎯 Type d'événement")

event_type = st.sidebar.selectbox(
    "Type d'événement",
    [
        "Tous",
        "Exposition",
        "Concert",
        "Atelier",
        "Visite guidée",
        "Conférence",
        "Activité enfants",
        "Famille",
        "Musique",
        "Science",
    ],
)

question = st.text_input(
    "Votre recherche",
    placeholder="Exemple : Je cherche une exposition scientifique",
)

if st.button("Rechercher"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
        st.stop()

    final_query = question.strip()

    if event_type != "Tous":
        final_query += f" de type {event_type}"

    if mode_localisation == "Choisir une ville":
        coordinates = get_city_coordinates(df, selected_city)

        if coordinates:
            user_lat, user_lon = coordinates
            final_query += f" autour de {selected_city}"
        else:
            st.warning("Coordonnées introuvables pour cette ville.")

    elif mode_localisation == "Saisir une adresse complète":
        if address and address.strip():
            coordinates = geocode_address(address)

            if coordinates:
                user_lat, user_lon = coordinates
                final_query += f" autour de l'adresse {address}"
            else:
                st.error("Adresse introuvable. Essayez une adresse plus précise.")
                st.stop()
        else:
            st.warning("Veuillez saisir une adresse complète.")
            st.stop()

    with st.spinner("Recherche en cours..."):
        context, docs = build_context(
            query=final_query,
            user_lat=user_lat,
            user_lon=user_lon,
            radius_km=radius_km
            if mode_localisation != "Toute l'Île-de-France"
            else None,
        )

        if not docs:
            st.warning("Aucun événement trouvé dans ce périmètre.")
            st.stop()

        answer = generate_answer(final_query, context)

    st.subheader("Question enrichie")
    st.write(final_query)

    if mode_localisation != "Toute l'Île-de-France":
        st.write(f"Rayon appliqué : {radius_km} km")

    st.subheader("Réponse")
    st.write(answer)