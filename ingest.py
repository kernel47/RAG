import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings
from rich import print
import shutil

DATA_DIR = "data"
DB_DIR = "db"

def ingest():
    print("[bold cyan]📦 Démarrage de l'indexation...[/bold cyan]")

    # Nettoyage de l’index si besoin
    if os.path.exists(DB_DIR):
        print("[yellow]⚠️ Suppression de l'index précédent...[/yellow]")
        shutil.rmtree(DB_DIR)

    # Initialiser embeddings offline
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Lire tous les fichiers
    loader = SimpleDirectoryReader(DATA_DIR, recursive=True, required_exts=[".txt", ".log", ".json", ".csv"])
    documents = loader.load_data()
    print(f"[green]✅ {len(documents)} documents chargés.[/green]")

    # Stockage vectoriel avec Chroma
    chroma_store = ChromaVectorStore(
        persist_dir=DB_DIR,
        chroma_settings=Settings(anonymized_telemetry=False)
    )
    storage_context = StorageContext.from_defaults(vector_store=chroma_store)

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
    index.storage_context.persist()

    print("[bold green]✅ Indexation terminée et sauvegardée.[/bold green]")

if __name__ == "__main__":
    ingest()
