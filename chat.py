import typer
from rich import print
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llm_wrapper_cpp import LlamaCppLLM  # ou ton wrapper ctransformers si tu veux revenir en arrière

DB_DIR = "db_faiss"
FAISS_INDEX_PATH = f"{DB_DIR}/faiss.index"
DOCSTORE_PATH = f"{DB_DIR}/docstore.json"

app = typer.Typer()

@app.command()
def chat():
    print("[bold cyan]💬 Assistant RAG FAISS (offline)[/bold cyan]")
    print("[yellow]Tape 'exit' pour quitter[/yellow]\n")

    # LLM local
    llm = LlamaCppLLM(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        n_batch=8,
        temperature=0.1,
        max_tokens=512,
    )

    # Embedding
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chargement de FAISS
    vector_store = FaissVectorStore(
        faiss_index_path=FAISS_INDEX_PATH,
        docstore_path=DOCSTORE_PATH,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        service_context=service_context,
    )

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)

    while True:
        q = input("[bold green]🔹 Question > [/bold green]").strip()
        if q.lower() in ["exit", "quit"]:
            print("[bold red]👋 Fin du chat.[/bold red]")
            break
        response = chat_engine.chat(q)
        print(f"\n[bold white]🧠 Réponse :[/bold white] {response.response}\n")

if __name__ == "__main__":
    app()
