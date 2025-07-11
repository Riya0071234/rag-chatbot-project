# 🤖 RAG Chatbot for User Agreement Analysis

This repository contains the source code for an AI-powered chatbot designed to answer user queries based on a provided user agreement document. The project implements a complete Retrieval-Augmented Generation (RAG) pipeline using a vector database and a real-time streaming interface built with Streamlit. The solution integrates either an open-source LLM via Ollama or OpenAI's GPT models.

---

## 🎥 Demo

[▶️ Click here to watch the screen recording of the chatbot in action](https://www.loom.com/share/9a20076c6b3b4f09a090a232fa73fbbf?sid=1b6d5bbe-f0d5-45e6-bbb6-2fff938a3a7d)
[▶️ Click here to watch the screen recording of the chatbot in action](https://www.loom.com/share/73a92d2c5f35497399343913036652a9?sid=4c51139a-8e3e-4027-a4a3-e6f73d849db4)


---

## 🏛️ Project Architecture

The application is built upon a classic RAG pipeline, which separates the **data ingestion** (offline) from **query processing** (online):

- **Data Ingestion (Offline)**:
  - Loads and parses the source PDF.
  - Chunks the content into semantically meaningful segments.
  - Embeds the chunks using `all-MiniLM-L6-v2`.
  - Stores them in a persistent **ChromaDB** vector database.

- **Query Processing (Online)**:
  - Embeds the user query and performs a similarity search on ChromaDB.
  - Injects the top relevant chunks into a prompt template.
  - Sends the prompt to a language model (OpenAI GPT or Ollama-hosted model).
  - Streams the final, grounded answer with sources back to the user in real-time.

---

## 📁 Folder Structure

├── app.py # Streamlit frontend
├── assets/ # Images or screenshots (optional)
├── data/
│ └── pdfs/ # Input PDF documents
├── notebooks/ # Evaluation & testing notebooks
├── prompts/
│ └── smart_qa.prompt # Prompt template for LLM
├── src/
│ ├── config.py
│ ├── embedding_wrapper.py
│ └── vector_db.py
├── data_ingestion.py # Offline pipeline
├── query_engine.py # CLI-based interface
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore config
└── README.md # This file



---

## ⚙️ Setup and Installation

### ✅ Prerequisites

- Python 3.9+
- `.env` file with the following:
OPENAI_API_KEY=your_openai_key_here



---

### 🧪 1. Clone the Repository

```bash
git clone (https://github.com/Riya0071234/rag-chatbot-project).git
cd rag-chatbot-project
🧪 2. Create and Activate a Virtual Environment
bash

python -m venv rag_env
# On Windows:
.\rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
📦 3. Install Dependencies
bash

pip install -r requirements.txt
🧠 4. Ingest the Document
bash

python data_ingestion.py
This processes the document, chunks it, embeds it, and stores vectors in ChromaDB.

🚀 5. Launch the Chatbot
bash

streamlit run app.py
This starts the Streamlit web interface at http://localhost:8501.

💡 Model & Tech Choices
Component	Choice	Reason
Embedding Model	all-MiniLM-L6-v2	Fast and semantically rich
Vector DB	ChromaDB	Simple, persistent, and local
LLM	 Ollama	Supports open-source LLMs locally
Prompting	External .prompt file	Keeps logic modular and editable

❓ Sample Queries
Here are some example queries you can test in the chatbot:

"What happens if I fail to pay for an item I purchase?"

"Can I use a robot or scraper to access the services?"

"What is the return policy for electronics?" (Failure case)

🧠 Notes on Alternatives
If you prefer local LLMs (e.g., mistral:instruct or gemma:2b):

bash

ollama pull mistral:instruct
# or
ollama pull gemma:2b
Update LLM_MODEL_NAME in src/config.py accordingly.

✅ .gitignore Highlights
rag_env/ (Python venv)

data/vectordb/ (Generated vectors)

__pycache__/

.env (Secrets)

📌 Project Deliverables
✅ Vector DB: data/vectordb/

✅ Preprocessed chunks: notebooks/

✅ RAG CLI: query_engine.py

✅ Streamlit App: app.py with streaming

✅ Demo Video: See Demo




