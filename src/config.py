# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Project Structure ---
# Dynamically determine the project root directory (assuming config.py is in 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data and Asset Paths ---
# Directory for storing the source PDF documents
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"

# Path to the specific document to be processed
# Update this if your PDF filename is different
SOURCE_PDF = PDF_DIR / "AI Training Document.pdf"

# Directory for storing the Chroma Vector Database
VECTOR_DB_DIR = DATA_DIR / "vectordb"

# Path to the prompt template file
PROMPT_FILE = PROJECT_ROOT / "prompts" / "smart_qa.prompt"

# --- Model Configurations ---

# 1. Embedding Model (used for vectorizing chunks and queries)
# 'all-MiniLM-L6-v2' is a fast and efficient choice.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 2. Large Language Model (LLM)
# This specifies the model Ollama should use (e.g., "mistral:instruct", "llama3")
# Ensure this model is pulled via Ollama CLI (e.g., ollama pull mistral:instruct)
LLM_MODEL_NAME = "mistral:instruct"
LLM_TEMPERATURE = 0.1  # Low temperature for factual, grounded answers

# --- Vector Database & Retrieval Parameters ---
CHROMA_COLLECTION_NAME = "rag_user_agreement"

# Chunking strategy: Aiming for the 100-300 word requirement.
# We measure in tokens for better precision with embedding models.
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

# Number of source chunks to retrieve during a search
RETRIEVER_K = 4

# --- Initialization ---
# Ensure necessary directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "prompts").mkdir(parents=True, exist_ok=True)