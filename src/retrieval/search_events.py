import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"


def load_vectorstore() -> FAISS:
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_events(query: str, k: int = 5) -> None:
    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )

    print(f"\nQuestion : {query}\n")
    print(f"{len(results)} événements trouvés :\n")

    for i, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        print("=" * 80)
        print(f"Résultat {i}")
        print(f"Score FAISS : {score}")
        print(f"Titre : {metadata.get('title')}")
        print(f"Ville : {metadata.get('city')}")
        print(f"Date : {metadata.get('start_date')}")
        print(f"URL : {metadata.get('url')}")
        print("-" * 80)
        print(doc.page_content[:700])
        print()


if __name__ == "__main__":
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY est absente du fichier .env")

    user_query = input("Pose ta question : ")
    search_events(user_query)