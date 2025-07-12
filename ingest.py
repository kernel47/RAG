import os
import shutil
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from rich import print

import faiss

DATA_DIR = "data"
DB_DIR = "db_faiss"

def ingest():
    print("[bold cyan]📦 Indexation avec FAISS...[/bold cyan]")

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Lecture des fichiers
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True, required_exts=[".txt", ".log", ".json", ".csv"]).load_data()
    print(f"[green]✅ {len(documents)} documents chargés[/green]")

    # FAISS vector store avec persist_path
    vector_store = FaissVectorStore(persist_path=DB_DIR)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    index.storage_context.persist(persist_dir=DB_DIR)

    print("[bold green]✅ Indexation persistée dans db_faiss/[/bold green]")

if __name__ == "__main__":
    ingest()
