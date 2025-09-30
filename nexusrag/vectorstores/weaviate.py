from typing import List, Dict, Any
import os
from .base import BaseVectorStore
from ..parsers.base import Document


class WeaviateVectorStore(BaseVectorStore):
    """Vector store implementation using Weaviate."""

    def __init__(self, class_name: str = "NexusRAGDocument", host: str = None):
        """Initialize the Weaviate vector store.

        Args:
            class_name (str): Name of the Weaviate class
            host (str): Weaviate host URL
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "To use WeaviateVectorStore, you need to install the weaviate-client library. "
                "Please run: pip install weaviate-client"
            )

        # Get host from environment variable or use default
        host = host or os.getenv("WEAVIATE_HOST", "http://localhost:8080")

        # Initialize Weaviate client
        self.client = weaviate.Client(host)
        self.class_name = class_name

        # Create class if it doesn't exist
        self._create_class_if_not_exists()

    def _create_class_if_not_exists(self):
        """Create the Weaviate class if it doesn't exist."""
        # Check if class exists
        if not self.client.schema.exists(self.class_name):
            # Define class schema
            class_schema = {
                "class": self.class_name,
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                    },
                    {
                        "name": "contentType",
                        "dataType": ["string"],
                    }
                ]
            }

            # Create class
            self.client.schema.create_class(class_schema)

    def add(self, docs: List[Document]) -> None:
        """Add documents to the Weaviate vector store.

        Args:
            docs (List[Document]): List of documents to add
        """
        # Add documents to Weaviate
        for doc in docs:
            data_object = {
                "content": doc.content,
                **doc.metadata
            }

            self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name
            )

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the Weaviate vector store for similar documents.

        Args:
            text (str): Query text
            top_k (int): Number of top results to return

        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        # Query Weaviate
        results = (
            self.client.query
            .get(self.class_name, ["content", "source", "contentType"])
            .with_near_text({"concepts": [text]})
            .with_limit(top_k)
            .do()
        )

        # Extract documents from results
        if (
            "data" in results and
            "Get" in results["data"] and
            self.class_name in results["data"]["Get"]
        ):
            documents = results["data"]["Get"][self.class_name]

            # Format results
            formatted_results = []
            for doc in documents:
                result = {
                    "content": doc.get("content", ""),
                    "metadata": {
                        "source": doc.get("source", ""),
                        "content_type": doc.get("contentType", "")
                    },
                    "score": 1.0  # Weaviate doesn't return scores in this query
                }
                formatted_results.append(result)

            return formatted_results

        return []
