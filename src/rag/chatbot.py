import os
from datetime import datetime

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"


def load_vectorstore():

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_context(query, k=5):

    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=k
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


def generate_answer(query):

    context = retrieve_context(query)

    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
Tu es un assistant spécialisé dans la recommandation d'événements culturels.

Date du jour : {today}

Tu dois répondre uniquement à partir des événements fournis dans le contexte.
Ne propose jamais un événement qui n'apparaît pas dans le contexte.
Si la question contient une contrainte temporelle comme "ce week-end", "aujourd'hui",
"demain" ou "cette semaine", utilise la date du jour pour l'interpréter.

Si aucun événement du contexte ne respecte la demande, réponds clairement :
"Aucun événement pertinent n'a été trouvé dans les données disponibles."

Contexte :
{context}

Question :
{query}

Réponse structurée :
"""

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2
    )

    response = llm.invoke(prompt)

    return response.content


def main():

    print("Assistant culturel RAG")
    print("Tape 'quit' pour quitter.\n")

    while True:

        query = input("Question : ")

        if query.lower() == "quit":
            break

        answer = generate_answer(query)

        print("\nRéponse :\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()