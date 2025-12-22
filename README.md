# 💁🏻CAMPUS FELLOW - Your personal university knowledge assistant 

# HOW IT WORKS
1. Document Understanding
Course PDFs are split into small chunks and converted into embeddings using
Sentence Transformers, capturing semantic meaning.

2. Vector Storage
All embeddings are stored in Qdrant, a vector database that enables fast
similarity-based retrieval with metadata support.

3. Context Retrieval
When a question is asked, the system retrieves the most relevant chunks
from the vector database based on semantic similarity.

4. Answer Generation
The retrieved context is passed to Groq’s LLaMA-3, which generates a
clear and grounded answer strictly based on the provided material.

