# 🎭 Assistant de Recommandation d'Événements Culturels basé sur un système RAG

## 📌 Contexte

Ce projet a été réalisé dans le cadre du **Projet 11 de la formation Data Engineer OpenClassrooms**.

L'objectif est de développer un assistant conversationnel capable de recommander des événements culturels à partir d'une requête formulée en langage naturel.

Le système s'appuie sur une architecture **RAG (Retrieval-Augmented Generation)** combinant :

* OpenAgenda pour la collecte des données ;
* Mistral Embed pour la vectorisation ;
* FAISS pour la recherche sémantique ;
* Mistral Small pour la génération des réponses ;
* LangChain pour l'orchestration ;
* Streamlit pour l'interface utilisateur.

---

# 🎯 Objectifs du projet

* Récupérer automatiquement les événements depuis l'API OpenAgenda.
* Filtrer les événements selon une zone géographique et une période donnée.
* Construire une base vectorielle permettant la recherche sémantique.
* Développer un chatbot capable de recommander des événements pertinents.
* Proposer une interface utilisateur simple via Streamlit.

---

# 🏗️ Architecture du projet

```text
OpenAgenda API
        │
        ▼
Prétraitement des données
        │
        ▼
Embeddings Mistral
        │
        ▼
Index vectoriel FAISS
        │
        ▼
Recherche sémantique
        │
        ▼
Mistral Small
        │
        ▼
Interface Streamlit
```

---

# 📂 Structure du projet

```text
projet11-rag-evenements/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── openagenda_events_raw.csv
│   │
│   └── processed/
│       └── openagenda_events_clean.csv
│
├── src/
│   ├── ingestion/
│   │   └── fetch_openagenda_events.py
│   │
│   ├── vectorization/
│   │   └── build_vectorstore.py
│   │
│   ├── retrieval/
│   │   └── search_events.py
│   │
│   ├── rag/
│   │   └── chatbot.py
│   │
│   ├── exploration/
│   │   ├── inspect_event.py
│   │   └── search_agendas.py
│   │
│   └── tests/
│       ├── test_events_dates.py
│       └── test_vectorstore.py
│
├── vectorstore/
│   └── faiss_index/
│
├── evaluation/
│   └── questions_reponses.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/ssignara/projet11-rag-evenements.git
cd projet11-rag-evenements
```

---

## 2. Créer l'environnement virtuel

Avec Conda :

```bash
conda create -n oc-p11-rag python=3.11
conda activate oc-p11-rag
```

---

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 4. Configurer les variables d'environnement

Créer un fichier `.env` :

```env
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_openagenda
```

---

# 📥 Collecte des données

Les données sont récupérées automatiquement depuis l'API OpenAgenda.

Lancer :

```bash
python src/ingestion/fetch_openagenda_events.py
```

Le script :

* récupère les événements ;
* filtre les événements de moins d'un an ;
* conserve uniquement l'Île-de-France ;
* enrichit les données avec les coordonnées GPS ;
* génère le dataset nettoyé.

Fichier généré :

```text
data/processed/openagenda_events_clean.csv
```

---

# 🔎 Construction de la base vectorielle

Lancer :

```bash
python src/vectorization/build_vectorstore.py
```

Ce script :

* charge les événements ;
* crée les embeddings avec Mistral Embed ;
* découpe les documents en chunks ;
* indexe les vecteurs dans FAISS.

Fichiers générés :

```text
vectorstore/faiss_index/index.faiss
vectorstore/faiss_index/index.pkl
```

---

# 🤖 Lancer le chatbot en ligne de commande

```bash
python src/rag/chatbot.py
```

Exemple :

```text
Question :
Je cherche une exposition scientifique à Paris
```

---

# 🌐 Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

Fonctionnalités :

* recherche en langage naturel ;
* filtrage par ville ;
* recherche à partir d'une adresse ;
* rayon kilométrique configurable ;
* filtrage par type d'événement.

---

# 🧪 Exécution des tests

Lancer :

```bash
pytest src/tests/
```

Tests réalisés :

* validation de la période des événements ;
* validation de la région géographique ;
* validation de l'existence de l'index FAISS.

Résultat attendu :

```text
3 passed
```

---

# 🧠 Modèles utilisés

## Modèle d'embedding

**mistral-embed**

Utilisé pour transformer les événements en vecteurs sémantiques afin d'effectuer des recherches par similarité.

### Pourquoi ce choix ?

* performant sur le benchmark MTEB ;
* compatible avec l'écosystème Mistral ;
* très adapté à la recherche sémantique.

---

## Modèle génératif

**mistral-small-latest**

Utilisé pour générer des réponses naturelles à partir des événements retrouvés dans FAISS.

### Pourquoi ce choix ?

* bon compromis entre performance et coût ;
* rapidité d'exécution ;
* intégration simple avec LangChain.

---

# 📊 Résultats obtenus

Dataset final :

```text
15 100 événements
```

Index vectoriel :

```text
15 100 documents
15 103 chunks
```

Tests :

```text
3 passed
```

---

# 🚀 Perspectives d'amélioration

* Géolocalisation automatique de l'utilisateur
* Mise à jour automatique de l'index
* Extension à l'ensemble du territoire français
* Historisation des préférences utilisateurs
* Déploiement Cloud (GCP, Azure ou AWS)

---

# 👤 Auteur

**Sokhna Signara Gueye**

Projet réalisé dans le cadre de la formation **Data Engineer OpenClassrooms**.
