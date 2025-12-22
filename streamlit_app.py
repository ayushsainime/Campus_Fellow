import streamlit as st
import requests
from PIL import Image
import time

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []



#  CONFIG 

BACKEND_URL = "http://127.0.0.1:8000/get_response"

st.set_page_config(
    page_title="Course RAG Assistant",
    page_icon="🎓",
    layout="centered"
)
# HELPER FUNCTIONS 
def stream_text(text, delay=0.01):
    """Typewriter-style animated text"""
    placeholder = st.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)

def load_image(path):
    try:
        return Image.open(path)
    except:
        return None


# ------------------ SIDEBAR ------------------

st.sidebar.title("💁🏻CampusFellow")

st.sidebar.markdown(
    """
**CampusFellow** is a Retrieval-Augmented Generation (RAG) based assistant  
that answers questions from **course notes, lectures, and syllabus**.
"""
)

st.sidebar.subheader("🛠️ Tech Stack")

# Load images (safe)

def load_image(path):
    try:
        return Image.open(path)
    except:
        return None

tech_images = [
    ("Python", "assets/Python_logo_and_wordmark.svg.png"),
    ("LangChain", "assets/langchain-color.png"),
    ("Qdrant", "assets/qdrant-logo-red-black.png"),
    ("Groq", "assets/Groq_logo.svg.png"),
    ("Streamlit", "assets/Streamlit-logo-primary-colormark-darktext.png"),
]

# Create grid: 2 columns per row
cols = st.sidebar.columns(2)

for idx, (tech, img_path) in enumerate(tech_images):
    col = cols[idx % 2]
    img = load_image(img_path)

    with col:
        if img:
            st.image(img, use_container_width=True)
        st.markdown(
            f"<p style='text-align: center; font-size: 12px;'><b>{tech}</b></p>",
            unsafe_allow_html=True
        )
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ How it Works")
st.sidebar.markdown(
    """
**1. Document Understanding**  
Course PDFs are split into small chunks and converted into embeddings using  
**Sentence Transformers**, capturing semantic meaning.

**2. Vector Storage**  
All embeddings are stored in **Qdrant**, a vector database that enables fast  
similarity-based retrieval with metadata support.

**3. Context Retrieval**  
When a question is asked, the system retrieves the most relevant chunks  
from the vector database based on semantic similarity.

**4. Answer Generation**  
The retrieved context is passed to **Groq’s LLaMA-3**, which generates a  
clear and grounded answer strictly based on the provided material.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("🔗 Links")
st.sidebar.markdown(
    """
- [GitHub Repository](https://github.com/ayushsainime)
- [LangChain](https://www.langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [Groq](https://groq.com/)
"""
)

st.sidebar.markdown(
    "<small>Built by - Ayush Saini</small>",
    unsafe_allow_html=True
)


#  UI HEADER 

st.title("💁🏻CampusFellow - Your personal university knowledge assistant")
st.markdown(
    """
Ever wondered why chatgpt doesn't know when is your course registration deadline  , 
well this **CHATBOT** does .\n

Ask questions from your own **course notes, lectures, and syllabus**.  
Answers are **grounded in your uploaded material**.
"""
)

#  USER INPUT 

query = st.text_input(
    "Enter your question",
    placeholder="e.g. Explain backpropagation in simple terms",
)

ask_button = st.button("Ask")

#  QUERY HANDLER 

if ask_button and query.strip():

    with st.spinner("🔍Searching course material and generating answer..."):
        try:
            response = requests.post(
                BACKEND_URL,
                data={"query": query},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Save to chat history
                st.session_state.chat_history.append(
                    {
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    }
                )

            else:
                st.error(f"Backend error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error("❌ Could not connect to backend.")
            st.caption(str(e))

elif ask_button:
    st.warning("Please enter a question.")

#  CHAT HISTORY DISPLAY 

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Conversation")

    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"### 🧠 Question ")
        st.write( chat["question"] )

        st.markdown("**📘 Answer:**")
        stream_text(chat["answer"])

        with st.expander("📄 Sources"):
            for src in chat["sources"]:
                st.write(src["content"])
                st.caption(f"Metadata: {src['metadata']}")
