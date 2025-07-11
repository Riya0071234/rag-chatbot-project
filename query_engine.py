# query_engine.py

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import configurations and helper classes from the 'src' directory
from src.config import (
    VECTOR_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    PROMPT_FILE,
    RETRIEVER_K
)
from src.vector_db import VectorDB
from src.embedding_wrapper import EmbeddingWrapper


def initialize_components():
    """
    Initializes and returns the core components for the RAG pipeline.

    Returns:
        A tuple containing the retriever and the LLM.
    """
    print("▶️ Initializing components...")
    embedding_model = EmbeddingWrapper(model_name=EMBEDDING_MODEL_NAME)
    vector_db = VectorDB(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model
    )
    retriever = vector_db.get_retriever(k=RETRIEVER_K)

    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    print("✅ Components initialized.")
    return retriever, llm


def format_docs(docs):
    """
    Formats the retrieved documents to include all relevant metadata for the prompt.

    Args:
        docs (list): A list of LangChain Document objects.

    Returns:
        str: A single string containing the formatted context.
    """
    return "\n\n---\n\n".join(
        f"Source: Title: {doc.metadata.get('title', 'N/A')}, Section: {doc.metadata.get('section', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}, Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def create_rag_chain(retriever, llm):
    """
    Creates the full RAG chain using the initialized components.

    Args:
        retriever: The document retriever.
        llm: The language model.

    Returns:
        The runnable RAG chain.
    """
    try:
        prompt_template = PROMPT_FILE.read_text()
        prompt = ChatPromptTemplate.from_template(prompt_template)
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at {PROMPT_FILE}")
        raise

    # The RAG chain, defined using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    """
    The main function to run a command-line interface for the RAG bot.
    """
    try:
        retriever, llm = initialize_components()
        rag_chain = create_rag_chain(retriever, llm)

        print(f"\n🧠 RAG Query Engine Ready! Using model: {LLM_MODEL_NAME}")
        print("Type your question and press Enter. Type 'exit' to quit.\n")

        while True:
            question = input("❓ Your question: ")
            if question.lower() == "exit":
                print("👋 Exiting...")
                break

            print("\n💬 Answer:")
            # The .stream() method invokes the chain and handles the streaming callback
            rag_chain.stream(question)
            print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"\n🔴 An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()