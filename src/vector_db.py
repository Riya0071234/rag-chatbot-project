# src/vector_db.py
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any

from src.embedding_wrapper import EmbeddingWrapper


class VectorDB:
    """
    A manager class for handling all ChromaDB vector store operations.

    This class encapsulates the logic for initializing, adding to, and retrieving
    from a persistent ChromaDB collection, ensuring consistent interaction
    with the vector database across the application.
    """
    def __init__(self, persist_directory: str, collection_name: str, embedding_function: EmbeddingWrapper):
        """
        Initializes the VectorDB manager.

        Args:
            persist_directory (str): The directory to save the persistent database.
            collection_name (str): The name of the collection to use.
            embedding_function (EmbeddingWrapper): The embedding model wrapper.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        try:
            # Initialize the persistent client
            self.client = PersistentClient(path=self.persist_directory)

            # Get or create the collection with cosine distance for semantic similarity
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB. Error: {e}")
            raise

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Adds documents to the ChromaDB collection.

        Chroma will automatically use the embedding_function provided during
        initialization to convert documents into vectors.

        Args:
            ids (List[str]): A list of unique identifiers for the documents.
            documents (List[str]): The text content of the documents.
            metadatas (List[Dict[str, Any]]): A list of metadata dictionaries.
        """
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ Successfully added {len(documents)} documents to the '{self.collection_name}' collection.")
        except Exception as e:
            print(f"❌ Failed to add documents to Chroma. Error: {e}")
            raise

    def get_retriever(self, k: int):
        """
        Creates a LangChain-compatible retriever from the vector store.

        Args:
            k (int): The number of top documents to retrieve for a query.

        Returns:
            A LangChain retriever object.
        """
        # Create a LangChain Chroma object from the existing client and collection
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )

        # Convert the vector store into a retriever
        return langchain_chroma.as_retriever(search_kwargs={"k": k})