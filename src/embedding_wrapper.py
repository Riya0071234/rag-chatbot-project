# src/embedding_wrapper.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

# A type hint for clarity, representing a list of floats for an embedding vector.
EmbeddingVector = List[float]

class EmbeddingWrapper:
    """
    A consistent wrapper for the HuggingFace embedding model.

    This class ensures that the same embedding model and its configurations
    are used throughout the application, from data ingestion to querying. It
    provides methods that are compatible with both LangChain and ChromaDB.
    """
    def __init__(self, model_name: str):
        """
        Initializes the embedding model wrapper.

        Args:
            model_name (str): The name of the HuggingFace model to use
                              (e.g., 'all-MiniLM-L6-v2').
        """
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Or 'cuda' if you have a GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"❌ Failed to load embedding model '{model_name}'. Error: {e}")
            raise

    def __call__(self, input: List[str]) -> List[EmbeddingVector]:
        """
        Makes the class callable, required by ChromaDB for embedding documents.

        Args:
            input (List[str]): A list of texts to embed.

        Returns:
            List[EmbeddingVector]: A list of embedding vectors.
        """
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Embeds a list of documents. This method is compatible with LangChain.

        Args:
            texts (List[str]): The documents to embed.

        Returns:
            List[EmbeddingVector]: The list of embedding vectors.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        """
        Embeds a single query. This method is compatible with LangChain retrievers.

        Args:
            text (str): The query text to embed.

        Returns:
            EmbeddingVector: The embedding vector for the query.
        """
        return self.model.embed_query(text)