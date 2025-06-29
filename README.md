# 📜 AI Policy Maker & Simulator
### A Retrieval-Augmented Generation (RAG) chatbot for policy analysis, simulations, and economic forecasting.  
🚀 Uses FAISS for vector search, EasyOCR for scanned PDFs, and Llama 3.2 for intelligent answers!

---

## 📌 Features
✅ **Upload Policy Documents (PDFs)** – Extracts **text** and **tables**  
✅ **OCR Support** – Reads **scanned PDFs** using **EasyOCR**  
✅ **FAISS Vector Search** – Retrieves **relevant policy data**  
✅ **Hallucination-Free Answers** – Uses **95% retrieved knowledge** & **5% AI reasoning**  
✅ **Simulations & Forecasting** – Supports **"What-if" policy analysis**  

---

## 📂 Project Structure
📁 AI-Policy-Simulator/ │── 📜 README.md <- Project documentation <br>
│── 📜 requirements.txt <- Python dependencies <br>
│── 📜 policy_simulator.py <- Main Streamlit app <br>
│── 📁 uploaded_pdfs/ <- Stores uploaded PDFs

---

## 🔧 Installation
### 1️⃣ Clone the Repository
```
git clone https://github.com/your-username/AI-Policy-Simulator.git
cd AI-Policy-Simulator
```
### 2️⃣ Setting up the Virtual Environment
```
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.\.venv\Scripts\activate   # Windows
```
### 3️⃣ Install the requirements.txt
```
pip install -r requirements.txt
```
### 4️⃣ Run .py using Streamlit
```
streamlit run policy_simulator.py
// Open http://localhost:8501 in your browser.
```

### Upload a policy document (PDF) and start asking questions!

### 📊 Example Questions You Can Ask
- "What is the total healthcare budget in 2024?"
- "If defense spending is cut by 5%, what impact will it have on GDP?"
- "Compare energy subsidies between 2020 and 2023."

### 🛠️ Technologies Used
- Python 3.10+ – Core language
- Streamlit – Interactive chatbot UI
- FAISS – Fast vector search for RAG
- EasyOCR – Extracts text from scanned PDFs
- Ollama – Uses Llama 3.2 for policy Q&A
- Sentence Transformers – thenlper/gte-large for embeddings
---
