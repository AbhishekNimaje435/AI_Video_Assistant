import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =====================================================
# Configuration
# =====================================================

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL =  "sentence-transformers/paraphrase-MiniLM-L3-v2"


# =====================================================
# Embeddings
# =====================================================

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# =====================================================
# Build Vector Store
# =====================================================

def build_vector_store(transcript: str):
    print("Building Vector Store...")

    # Remove old DB to avoid stale embeddings
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_text(transcript)

    print(f"Total Chunks: {len(chunks)}")

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(
        f"Vector Store Created Successfully ({vector_store._collection.count()} chunks)"
    )

    return vector_store


# =====================================================
# Load Existing Vector Store
# =====================================================

def load_vector_store():
    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# =====================================================
# Retriever
# =====================================================

def get_retriever(vector_store, k=6):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )