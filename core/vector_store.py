import os
import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = str(Path(tempfile.gettempdir()) / "ai_video_assistant" / "vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(transcript: str):
    os.makedirs(os.path.dirname(CHROMA_DIR), exist_ok=True)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_text(transcript)

    if not chunks:
        raise ValueError("Transcript is empty; cannot build the vector store.")

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store created: {len(docs)} chunks")
    return vector_store


def load_vector_store():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError("Vector store does not exist yet.")

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store, k=6):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
