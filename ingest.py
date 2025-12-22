import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams



# ------------------ CONFIG ------------------

DATA_PATH = "Data"   # PDFs: lecture notes, syllabus, assignments, etc.
COLLECTION_NAME = "course_rag_vectors"
QDRANT_URL = "http://localhost:6333"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 output size


# ------------------ INGEST STEPS ------------------

def load_documents():
    """
    Loads all course-related PDFs (lectures, syllabus, notes).
    """
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    return loader.load()


def split_documents(documents):
    """
    Splits documents into overlapping chunks
    to preserve academic context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def create_embeddings():
    """
    Creates embeddings using a general-purpose
    academic-quality embedding model.
    """
    return SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )


def ingest():
    print("Loading course documents")
    documents = load_documents()

    print("Splitting documents into chunks")
    chunks = split_documents(documents)

    print(" Creating embeddings model")
    embeddings = create_embeddings()

    print(" Connecting to Qdrant")
    client = QdrantClient(url=QDRANT_URL)

    # ---- Explicit collection creation (safe & idempotent) ----
    existing_collections = [
        col.name for col in client.get_collections().collections
    ]

    if COLLECTION_NAME not in existing_collections:
        print(" Creating Qdrant collection")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(" Qdrant collection already exists")

    print(" Creating vector store interface")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print(" Adding course documents to vector store")
    vectorstore.add_documents(chunks)

    print("Course knowledge vector database successfully created!")


if __name__ == "__main__":
    ingest()
