import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_groq import ChatGroq


#  FASTAPI SETUP 

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


#  CONFIG 

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "course_rag_vectors"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Groq model options:
# "llama3-8b-8192"   → fast & cheap
# "llama3-70b-8192"  → best quality
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

from secret import GROQ_API_KEY

groq_api = GROQ_API_KEY
#  EMBEDDINGS 

embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


#  VECTOR STORE 


client = QdrantClient(url=QDRANT_URL)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print(" Vector store & retriever ready")


#  GROQ LLM 


llm = ChatGroq(
    groq_api_key = groq_api,
    model=GROQ_MODEL_NAME,
    temperature=0.2,
)

print("Groq LLM initialized")


#  PROMPT 

PROMPT_TEMPLATE = """
You are a helpful university course assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"I don't know based on the provided course material. "

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


#  CONTEXT FORMATTER 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#  LCEL RAG CHAIN 

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


#  ROUTES 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Retrieve source documents explicitly
    docs = retriever.invoke(query)

    # Generate answer
    answer = rag_chain.invoke(query)

    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    return JSONResponse(
        content={
            "answer": answer,
            "sources": sources,
        }
    )
