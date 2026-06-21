import os

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()

DATASET_PATH = "data/processed/openagenda_events_clean.csv"
VECTORSTORE_PATH = "vectorstore/faiss_index"


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé.
    """
    return pd.read_csv(DATASET_PATH)


def create_documents(df: pd.DataFrame) -> list[Document]:
    """
    Transforme chaque événement en Document LangChain.
    """
    documents = []

    for _, row in df.iterrows():

        metadata = {
            "event_id": row.get("Identifiant"),
            "title": row.get("Titre"),
            "city": row.get("Ville"),
            "department": row.get("Département"),
            "region": row.get("Région"),
            "start_date": row.get("Première date - Début"),
            "end_date": row.get("Dernière date - Fin"),
            "url": row.get("URL canonique"),
            "latitude": row.get("Latitude"),
            "longitude": row.get("Longitude"),
        }

        documents.append(
            Document(
                page_content=row["text_for_embedding"],
                metadata=metadata,
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    return splitter.split_documents(documents)


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Génère les embeddings et construit l'index FAISS.
    """
    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Sauvegarde l'index FAISS.
    """
    os.makedirs("vectorstore", exist_ok=True)

    vectorstore.save_local(
        VECTORSTORE_PATH
    )


def main() -> None:

    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "MISTRAL_API_KEY absente du fichier .env"
        )

    df = load_dataset()

    print(f"Dataset chargé : {len(df)} événements")

    documents = create_documents(df)

    print(f"Documents créés : {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks créés : {len(chunks)}")

    vectorstore = build_vectorstore(chunks)

    save_vectorstore(vectorstore)

    print(
        f"Index FAISS sauvegardé dans : {VECTORSTORE_PATH}"
    )


if __name__ == "__main__":
    main()