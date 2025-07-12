import typer
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llm_wrapper import CTransformersLLM
from rich import print

DB_DIR = "db"
app = typer.Typer()

@app.command()
def chat():
    print("[bold cyan]💬 Assistant RAG Backup (offline) prêt[/bold cyan]")
    print("[bold yellow]Tape 'exit' pour quitter[/bold yellow]\n")

    # Charger le LLM local via ctransformers
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # adapte ce chemin selon ta config
    llm = CTransformersLLM(model_path=model_path, temperature=0.1)

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    storage_context = StorageContext.from_defaults(persist_dir=DB_DIR)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    index = load_index_from_storage(storage_context, service_context=service_context)

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
