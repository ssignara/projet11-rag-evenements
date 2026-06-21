# Projet 11 — Assistant de recommandation d'événements culturels

## Objectif du projet

Ce projet consiste à développer un Proof of Concept d'assistant conversationnel capable de recommander des événements culturels en Île-de-France à partir de données OpenAgenda.

Le système repose sur une architecture RAG : Retrieval-Augmented Generation.

## Fonctionnalités

- Récupération des événements via l'API OpenAgenda
- Filtrage des événements de moins d'un an
- Nettoyage et structuration des données
- Création d'un champ texte enrichi pour la vectorisation
- Génération d'embeddings avec Mistral
- Indexation dans FAISS
- Recherche sémantique avec LangChain
- Génération de réponses naturelles avec Mistral
- Interface Streamlit

## Architecture

```text
API OpenAgenda
      ↓
Prétraitement Pandas
      ↓
Dataset propre CSV
      ↓
Documents LangChain
      ↓
Embeddings Mistral
      ↓
Index FAISS
      ↓
Recherche sémantique
      ↓
Réponse générée par Mistral
      ↓
Interface Streamlit
Installation

Créer l'environnement :

conda create -n oc-p11-rag python=3.11 -y
conda activate oc-p11-rag

Installer les dépendances :

pip install -r requirements.txt

Créer un fichier .env :

OPENAGENDA_API_KEY=votre_cle_openagenda
OPENAGENDA_AGENDA_UID=56500817
MISTRAL_API_KEY=votre_cle_mistral
Exécution du pipeline
1. Récupérer et nettoyer les données
python src/ingestion/fetch_openagenda_events.py
2. Construire l'index FAISS
python src/vectorization/build_vectorstore.py
3. Tester la recherche sémantique
python src/retrieval/search_events.py
4. Lancer le chatbot RAG
python src/rag/chatbot.py
5. Lancer l'application Streamlit
streamlit run app/streamlit_app.py
Tests

Lancer les tests unitaires :

pytest src/tests/

Les tests vérifient :

que les événements ont moins d'un an ;
que l'index FAISS est bien généré.
Données utilisées

Les données proviennent de l'API OpenAgenda, via l'agenda officiel :

OpenAgenda en Île-de-France
UID : 56500817

Le périmètre retenu est l'Île-de-France.

Modèles utilisés
Embeddings : mistral-embed
LLM : mistral-small-latest
Base vectorielle : FAISS
Orchestration : LangChain
Évaluation

Un jeu de test annoté est disponible dans :

evaluation/questions_reponses.csv

Il contient des questions représentatives et les réponses attendues pour évaluer la pertinence du système.

Limites du POC
Le filtrage géographique avancé par rayon kilométrique est préparé via les coordonnées GPS, mais peut être amélioré.
Les performances dépendent de la qualité des descriptions OpenAgenda.
Le système ne conserve pas d'historique conversationnel.
Le POC est limité à l'Île-de-France.
Améliorations possibles
Ajouter un calcul de distance utilisateur/événement.
Ajouter des filtres par date, catégorie, prix ou accessibilité.
Déployer l'application sur le cloud.
Mettre à jour automatiquement l'index FAISS.
Évaluer quantitativement les réponses avec un score de similarité.