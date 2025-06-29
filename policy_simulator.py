import os
import time
import streamlit as st
import ollama
import faiss
import numpy as np
import easyocr
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# 📌 Load Embedding Model (Optimized for policy & finance data)
embedder = SentenceTransformer("thenlper/gte-large")

# 📌 Initialize OCR Model for Scanned PDFs
ocr_reader = easyocr.Reader(["en"])

# 📌 Load & Extract Text from PDFs (With OCR for Scanned PDFs)
def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    extracted_text = "\n".join([page.page_content for page in pages])

    # ✅ Debugging: Print Extracted Text
    print("📌 Extracted Text from PDF:\n", extracted_text[:2000])

    # 📌 Apply OCR if no text was extracted
    if not extracted_text.strip():
        print("⚠️ No text found! Applying OCR...")
        extracted_text = extract_text_with_ocr(pdf_path)

    return pages, extracted_text

# 📌 OCR for Scanned PDFs
def extract_text_with_ocr(pdf_path):
    result = ocr_reader.readtext(pdf_path, detail=0, paragraph=True)
    extracted_text = "\n".join(result)

    print("📌 Extracted OCR Text:\n", extracted_text[:2000])
    return extracted_text

# 📌 Split Text into Chunks for FAISS Indexing
def split_documents(text_data, chunk_size=4096, chunk_overlap=500):
    """Manually splits text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text_data), chunk_size - chunk_overlap):
        chunks.append(text_data[i:i+chunk_size])
    return chunks

# 📌 Create FAISS Vector Store
def create_vector_store(split_docs):
    if not split_docs:
        print("⚠️ No valid documents found for embedding.")
        return None, None, None

    embeddings = embedder.encode(split_docs)
    dimension = embeddings.shape[1]

    # Initialize FAISS Index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    return index, split_docs, embedder

# 📌 Retrieve Relevant Context from FAISS
def retrieve_context(query, embedder, index, documents, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0] if i < len(documents)]

# 📌 **Enhanced Prompt for Strict Retrieval-Based Answers**
def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context) if context else "No relevant policy data found."

    policy_prompt = f"""
    **You are a Government Policy & Economic Impact AI Advisor.**
    
    📌 **Guidelines:**
    - **Use 95% retrieved document knowledge** and **limit AI enhancement to 5%**.
    - **You cannot hallucinate** or fabricate missing policy details.
    - **If data is insufficient**, respond: "The document does not contain this information."

    --- 
    📊 **Capabilities:**
    - Analyze **government budgets, taxation, and economic policies**.
    - Simulate **GDP, inflation, and job market impact** of policy changes.
    - Compare past and current **budget allocations & fiscal policies**.

    --- 
    **Policy Data Retrieved from Documents (95% Weightage):**
    {formatted_context}

    **User's Question:** {query}
    
    🔹 Provide a **structured, data-driven response** using only the provided policy document.  
    🔹 **Limit AI enhancement to 5%** for language clarity and structure only.  
    🔹 **DO NOT** generate speculative answers.  
    """

    # 🚀 Call Ollama with Llama 3.2 Model & Enhanced Prompt
    response = ollama.chat(
        model="captain-corgi/corgi:latest",
        messages=[
            {"role": "system", "content": "You are a strict policy analyst. Do NOT generate extra details beyond retrieved documents."},
            {"role": "user", "content": policy_prompt},
        ],
        options={"temperature": 0.1, "max_tokens": 4096},  # **Lowered temperature to reduce hallucination**
    )
    return response["message"]["content"]


# 📌 Typing Effect for Realistic Response
def typing_effect(text, delay=0.05):
    typed_text = ""
    placeholder = st.empty()
    for char in text:
        typed_text += char
        placeholder.markdown(f"**Answer:** {typed_text}")
        time.sleep(delay)

# 📌 Streamlit UI Setup
st.title("📜 AI Policy Maker - Llama 3.2 (Ollama)")

# 📌 Upload Policy PDFs
st.sidebar.header("Upload Policy Reports")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing Policy Documents..."):
        pdf_path = "./uploads/policy_reports.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages, extracted_text = load_pdf_documents(pdf_path)
        split_docs = split_documents(extracted_text)
        index, document_texts, embedder = create_vector_store(split_docs)

        st.session_state["index"] = index
        st.session_state["documents"] = document_texts
        st.session_state["embedder"] = embedder
        st.success("✅ Policy reports uploaded & processed successfully!")

        # ✅ Show extracted text preview in UI
        st.subheader("📜 Extracted Text Preview")
        st.text_area("Extracted Text:", extracted_text[:2000], height=300)

# 📌 Chat Interface
st.subheader("Ask Policy & Economic Questions")

query = st.text_input("Ask a policy simulation question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            index = st.session_state.get("index")
            if not index:
                st.error("Please upload policy reports first!")
            else:
                context = retrieve_context(query, st.session_state["embedder"], st.session_state["index"], st.session_state["documents"])
                answer = generate_answer_with_ollama(query, context)
                typing_effect(answer)
    else:
        st.warning("Please enter a question.")

st.sidebar.write("📊 Built with Streamlit, Ollama, FAISS, EasyOCR & Policy Reports")
