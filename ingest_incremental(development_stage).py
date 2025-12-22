import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


DATA_PATH = "Data"
COLLECTION_NAME = "course_rag_vectors"
QDRANT_URL = "http://localhost:6333"
TRACK_FILE = "ingested_files.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_tracked_files():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_tracked_files(files):
    with open(TRACK_FILE, "w") as f:
        json.dump(list(files), f, indent=2)


def ingest_incremental():
    tracked_files = load_tracked_files()

    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = QdrantClient(url=QDRANT_URL)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    new_files = []

    for filename in os.listdir(DATA_PATH):
        if not filename.endswith(".pdf"):
            continue

        if filename in tracked_files:
            continue

        print(f"📄 New file detected: {filename}")

        loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
        documents = loader.load()

        chunks = splitter.split_documents(documents)
        vectorstore.add_documents(chunks)

        tracked_files.add(filename)
        new_files.append(filename)

    save_tracked_files(tracked_files)

    print(f"✅ Added {len(new_files)} new files to vector DB")


if __name__ == "__main__":
    ingest_incremental()
