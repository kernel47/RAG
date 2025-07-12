import os
import shutil
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from rich import print
import faiss

DATA_DIR = "data"
DB_DIR = "db_faiss"
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(DB_DIR, "docstore.json")

def ingest():
    print("[bold cyan]📦 Indexation avec FAISS...[/bold cyan]")

    # Nettoyage (optionnel)
    if os.path.exists(DB_DIR):
        print("[yellow]⚠️ Suppression ancienne base...[/yellow]")
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Documents
    loader = SimpleDirectoryReader(DATA_DIR, recursive=True, required_exts=[".txt", ".log", ".json", ".csv"])
    documents = loader.load_data()
    print(f"[green]✅ {len(documents)} documents chargés[/green]")

    # Stockage avec FAISS
    vector_store = FaissVectorStore(faiss_index_path=FAISS_INDEX_PATH, docstore_path=DOCSTORE_PATH)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    index.storage_context.persist(persist_dir=DB_DIR)

    print("[bold green]✅ Indexation FAISS terminée et persistée[/bold green]")

if __name__ == "__main__":
    ingest()
