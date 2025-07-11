# data_ingestion.py

import re
from pathlib import Path
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import configurations from the 'src' directory
from src.config import (
    SOURCE_PDF,
    VECTOR_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
)
from src.vector_db import VectorDB
from src.embedding_wrapper import EmbeddingWrapper


def load_and_parse_pdf(pdf_path: Path) -> list[Document]:
    """
    Loads a PDF, identifies section titles, and attaches rich metadata
    to the text content of each page.

    Args:
        pdf_path (Path): The path to the PDF file.

    Returns:
        list[Document]: A list of Document objects with detailed metadata.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF not found at the specified path: {pdf_path}")

    print(f"📄 Loading and parsing PDF: {pdf_path.name}")
    documents = []

    # Regex to identify section headers like "1. Introduction" or "19. Legal Disputes"
    section_pattern = re.compile(r"^\s*(\d{1,2}\.\s+[A-Z][A-Za-z\s&;’(),\-\/]+)")

    try:
        doc = fitz.open(pdf_path)
        doc_title = doc.metadata.get("title", pdf_path.stem)
        current_section = "Introduction"  # Default section before the first header is found

        for page_num, page in enumerate(doc):
            full_page_text = ""
            blocks = page.get_text("blocks")  # Get text in structured blocks

            for block in blocks:
                block_text = block[4].strip()  # The text content is the 5th element

                # Check if the block text matches our section header pattern
                match = section_pattern.match(block_text)
                if match:
                    current_section = match.group(1).strip()
                    # We might skip adding the header itself as a document
                    # or add it as a standalone piece of metadata.
                    # For simplicity, we'll just update the current section name.

                # We add the text to the page content and tag it with metadata
                # Note: A more advanced version could chunk here directly
                full_page_text += block_text + "\n"

            # Create one document per page with the last known section
            metadata = {
                "source_file": pdf_path.name,
                "page": page_num + 1,
                "title": doc_title,
                "section": current_section  # Note: this tags the whole page with the last section found
            }
            documents.append(Document(page_content=full_page_text, metadata=metadata))

        doc.close()
        print(f"✅ Successfully loaded and parsed {len(documents)} pages.")
        return documents
    except Exception as e:
        print(f"❌ Failed to process PDF. Error: {e}")
        raise


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits the loaded documents into smaller, manageable chunks.
    The metadata from the parent page (including section) is preserved.
    """
    print(f"🔪 Chunking documents... (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    # Assign a unique ID to each chunk for easier identification
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{Path(chunk.metadata['source_file']).stem}_chunk_{i}"

    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def initialize_and_ingest(chunks: list[Document]):
    """
    Initializes the embedding model and vector database, then ingests chunks.
    """
    print("🧠 Initializing embedding model and vector database...")
    embedding_model = EmbeddingWrapper(model_name=EMBEDDING_MODEL_NAME)
    vector_db = VectorDB(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model
    )

    print(f"💾 Ingesting {len(chunks)} chunks into ChromaDB...")
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vector_db.add(ids=ids, documents=documents, metadatas=metadatas)


def main():
    """The main function to run the entire data ingestion pipeline."""
    print("🚀 Starting data ingestion pipeline...")
    documents = load_and_parse_pdf(SOURCE_PDF)
    if documents:
        chunks = chunk_documents(documents)
        initialize_and_ingest(chunks)
        print("\n🎉 Data ingestion pipeline completed successfully!")
    else:
        print("🔴 Pipeline stopped: No documents were loaded from the PDF.")


if __name__ == "__main__":
    main()