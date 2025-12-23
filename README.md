# 💁🏻CampusFellow — Your Personal University Knowledge Assistant

CampusFellow is a Retrieval-Augmented Generation (RAG) based system that allows students to query their **own university course material**—lecture notes, syllabi, PDFs, and academic documents—using natural language.

Instead of searching through chats, PDFs, and folders, LectureLens provides **context-aware, source-grounded answers** powered by modern vector search and large language models.

---

## 🚀 Features

- 📚 Ask questions directly from your course notes and academic PDFs  
- 🔍 Semantic search using vector embeddings (not keyword-based search)  
- 🧠 Context-aware answers grounded in your own documents  
- 📄 Source visibility for transparency and verification  
- ⚡ Ultra-fast responses using Groq-powered LLaMA-3  
- 🖥️ Interactive Streamlit UI  
- 🐳 Fully Dockerized and reproducible setup  

<img width="1919" height="960" alt="Screenshot 2025-12-23 002417" src="https://github.com/user-attachments/assets/eab0ccb6-7265-433a-a5b7-8d977483956b" />

---

## 🧠 System Architecture

Course PDFs → Text Chunking → Sentence Transformer Embeddings → Qdrant Vector DB → Context Retrieval → Groq LLaMA-3 → FastAPI Backend → Streamlit Frontend

---

## 🛠️ Tech Stack

- **Python 3.11**
- **LangChain (LCEL / Runnable-based RAG)**
- **Sentence Transformers** (for embeddings)
- **Qdrant** (vector database)
- **Groq API** (LLaMA-3 inference)
- **FastAPI** (backend API)
- **Streamlit** (frontend UI)
- **Docker**

---
## ⚙️ Prerequisites

Before running the project, ensure you have:

- **Python 3.10 or 3.11**
- **Docker & Docker Compose**
- A **Groq API key**

Install Docker from:  

👉 https://www.docker.com/products/docker-desktop/

---

## 🔐 Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

# 🧩 Step 1: Add Your Course Data

Place your academic PDFs inside the Data/ folder:
Data/
 ├── syllabus.pdf
 ├── lecture_1.pdf
 ├── lecture_2.pdf
 ├── notes.pdf

# 🐳 STEP 2: Pull and Run Qdrant Using Docker

Pull the official Qdrant image:
```
docker pull qdrant/qdrant
```

Run Qdrant as a container with persistent storage:
```
docker run -p 6333:6333 ^
  -v qdrant_storage:/qdrant/storage ^
  qdrant/qdrant
```

Verify Qdrant is running by opening:
```
http://localhost:6333/dashboard
```

# 📥 STEP 3: Create Virtual Environment & Install Dependencies

Create and activate a virtual environment:
```
python -m venv .env
.env\Scripts\activate   # Windows
```

Install required Python packages:
```
pip install -r requirements.txt
```

# 🧠 STEP 4: Run Ingestion Script (One-Time)
Run:
```
python ingest.py
```

# ⚡ STEP 5: Run FastAPI Backend

Start the backend server:
```
uvicorn rag:app --reload
```

Backend will be available at:
```
http://127.0.0.1:8000
```

# 🎨 STEP 6: Run Streamlit Frontend

Open a new terminal, activate the same virtual environment, and run:
```
streamlit run streamlit_app.py
```

Streamlit UI will open at:
```
http://localhost:8501
```

# 🔄 Updating Course Content

If you add or modify PDFs:

Update files in the Data/ folder

Re-run:
```
python ingest.py
```

Restart FastAPI (Streamlit can stay running)
