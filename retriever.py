# app/retriever.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_index.schema import Document

EMBEDDINGS_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "./data/index.faiss"
DOCS_PATH = "./data/docs.pkl"

# Mémoire persistante : vecteurs + documents
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(384)  # 384 = dim all-MiniLM-L6-v2

if os.path.exists(DOCS_PATH):
    with open(DOCS_PATH, "rb") as f:
        stored_docs = pickle.load(f)
else:
    stored_docs = []


def add_documents(documents: list[Document]):
    texts = [doc.page_content for doc in documents]
    vectors = EMBEDDINGS_MODEL.encode(texts, show_progress_bar=True)
    index.add(vectors)
    stored_docs.extend(documents)

    # save
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(stored_docs, f)


def query_documents(query: str, top_k=5):
    vector = EMBEDDINGS_MODEL.encode([query])
    distances, indices = index.search(vector, top_k)
    return [stored_docs[i] for i in indices[0] if i < len(stored_docs)]
