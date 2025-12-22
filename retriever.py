from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient


# ------------------ CONFIG ------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "course_rag_vectors"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ------------------ SETUP ------------------

# Embedding model (MUST match ingestion)
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# Qdrant client
client = QdrantClient(
    url=QDRANT_URL
)

print("Connected to Qdrant")


# ------------------ LOAD VECTOR STORE ------------------

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

print("Vector store loaded successfully")


# ------------------ RETRIEVAL ------------------

query = "tell me about kaizen crew "

docs_with_scores = vectorstore.similarity_search_with_score(
    query=query,
    k=3
)

for doc, score in docs_with_scores:
    print({
        "score": score,
        "content": doc.page_content,
        "metadata": doc.metadata
    })
