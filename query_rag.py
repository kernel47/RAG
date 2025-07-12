# query_rag.py
import sys
from app.retriever import query_documents
from llama_cpp import Llama

# Charger le modèle LLaMA local
llm = Llama(model_path="./models/mistral-7b-instruct.Q4_K_M.gguf", n_ctx=4096, n_threads=8)


def build_prompt(query: str, contexts: list):
    context_text = "\n\n".join([doc.page_content for doc in contexts])
    return f"Tu es un assistant pour l'audit IT. Voici les données :\n\n{context_text}\n\nQuestion : {query}\nRéponse claire et concise :"


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_rag.py 'ta question ici'")
        return

    query = sys.argv[1]
    context_docs = query_documents(query)

    if not context_docs:
        print("Aucun contexte trouvé pour cette question.")
        return

    prompt = build_prompt(query, context_docs)
    output = llm(prompt, max_tokens=300, stop=["\n\n"])
    print("\n🧠 Réponse IA :")
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
