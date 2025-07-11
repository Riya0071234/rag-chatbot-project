import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(project_root))

# Import configurations and core RAG components from the 'src' directory
from src.config import (
    LLM_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
    RETRIEVER_K,
    PROMPT_FILE
)
from src.embedding_wrapper import EmbeddingWrapper
from src.vector_db import VectorDB
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Use Streamlit's caching to load the pipeline only once, improving performance.
@st.cache_resource
def load_rag_pipeline():
    """
    Initializes and caches the core components of the RAG pipeline.

    This function loads the embedding model, vector database, retriever, and LLM,
    and constructs the RAG chain. It's cached to prevent reloading on every
    user interaction.

    Returns:
        A tuple containing the retriever, the runnable RAG chain, and the
        number of documents in the vector store.
    """
    print("▶️ Loading RAG pipeline components...")
    embedding_model = EmbeddingWrapper(model_name=EMBEDDING_MODEL_NAME)
    vector_db = VectorDB(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model
    )
    doc_count = vector_db.collection.count()

    retriever = vector_db.get_retriever(k=RETRIEVER_K)
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)

    try:
        prompt_template = PROMPT_FILE.read_text()
        prompt = ChatPromptTemplate.from_template(prompt_template)
    except FileNotFoundError:
        st.error(f"Prompt file not found at {PROMPT_FILE}. Please ensure it exists.")
        return None, None, 0

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"Source: Title: {doc.metadata.get('title', 'N/A')}, Section: {doc.metadata.get('section', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print("✅ RAG pipeline loaded successfully.")
    return retriever, rag_chain, doc_count


def main():
    """
    The main function that defines and runs the Streamlit web application.
    """
    st.set_page_config(page_title="DocBot", page_icon="🤖", layout="wide")
    st.title("📄 AI Document Chatbot")

    # [cite_start]--- Sidebar Setup [cite: 462] ---
    with st.sidebar:
        st.header("⚙️ Controls & Info")

        # Load pipeline and get doc count for the sidebar
        try:
            retriever, rag_chain, doc_count = load_rag_pipeline()
        except Exception as e:
            st.error(f"Failed to load RAG pipeline: {e}")
            return


        st.info(f"**Model:** `{LLM_MODEL_NAME}` [cite: 463]")

        st.info(f"**Indexed Chunks:** `{doc_count}` [cite: 464]")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # [cite_start]User input field [cite: 459]
    if user_query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and stream the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve sources before generating the answer
                retrieved_sources = retriever.get_relevant_documents(user_query)

                # [cite_start]Stream the response from the RAG chain [cite: 460]
                response_stream = rag_chain.stream(user_query)
                full_response = st.write_stream(response_stream)

                # [cite_start]Display the sources in an expander [cite: 461]
                with st.expander("View Sources Used for This Answer"):
                    for i, doc in enumerate(retrieved_sources):
                        st.markdown(
                            f"**Source {i + 1}: Page {doc.metadata.get('page', 'N/A')} | Section: `{doc.metadata.get('section', 'N/A')}`**")
                        st.info(doc.page_content)

        st.session_state.messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()