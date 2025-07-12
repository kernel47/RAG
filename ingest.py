import os
import shutil
import faiss
from pathlib import Path
from rich import print
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

DATA_DIR = "data"
DB_DIR = "db_faiss"

def ingest():
    print("[bold cyan]📦 Indexation avec FAISS (custom) ...[/bold cyan]")

    # Nettoyage
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    # Embedding
    embed_model = HuggingFaceEmbedding(model_name="transformers/all-MiniLM-L6-v2")

    # Créer un index FAISS brut
    dim = 384  # Dimensions de all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatL2(dim)  # index L2 sans training

    # Construire le vector store
    vector_store = FaissVectorStore(
        faiss_index=faiss_index,
        docstore=None,  # sera généré automatiquement par LlamaIndex
        index_id="main_index"
    )

    # Stockage LlamaIndex
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    # Charger les documents
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True, required_exts=[".txt", ".log", ".json", ".csv"]).load_data()
    print(f"[green]✅ {len(documents)} documents chargés[/green]")

    # Construire l’index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    # Sauvegarde manuelle
    index.storage_context.persist(persist_dir=DB_DIR) 

    print("[bold green]✅ FAISS index sauvegardé dans db_faiss/[/bold green]")

if __name__ == "__main__":
    ingest()
