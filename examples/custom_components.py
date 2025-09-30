"""
Custom components example for NexusRAG.

This example demonstrates how to create and use custom components with the NexusRAG framework.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.pipeline import RAGPipeline
from nexusrag.parsers.base import BaseParser, Document
from nexusrag.embedders.base import BaseEmbedder
from nexusrag.vectorstores.base import BaseVectorStore
from nexusrag.llms.base import BaseLLM


class CustomParser(BaseParser):
    """A custom parser that creates simple document objects."""
    
    def parse(self, file_path: str) -> list:
        """Parse a text file and return a list of Document objects."""
        # For this example, we'll just create a simple document
        # In a real implementation, you would parse the actual file
        content = f"This is a document parsed from {file_path}. "
        content += "It contains some sample text for demonstration purposes. "
        content += "NexusRAG is a modular framework for building AI agents."
        
        document = Document(content=content, metadata={"source": file_path})
        return [document]


class CustomEmbedder(BaseEmbedder):
    """A custom embedder that creates simple embeddings."""
    
    def embed(self, texts: list) -> list:
        """Generate simple embeddings for a list of texts."""
        # For this example, we'll create simple embeddings based on text length
        # In a real implementation, you would use an actual embedding model
        embeddings = []
        for text in texts:
            # Create a simple embedding based on text characteristics
            embedding = [
                len(text) / 1000,  # Normalized length
                text.count(' ') / len(text) if len(text) > 0 else 0,  # Space density
                text.count('.') / len(text) if len(text) > 0 else 0,  # Period density
                text.count('a') / len(text) if len(text) > 0 else 0,  # 'a' frequency
                text.count('e') / len(text) if len(text) > 0 else 0,  # 'e' frequency
            ]
            embeddings.append(embedding)
        return embeddings


class CustomVectorStore(BaseVectorStore):
    """A custom vector store that stores documents in memory."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.documents = []
        self.embeddings = []
    
    def add(self, docs: list) -> None:
        """Add documents to the vector store."""
        # For this example, we'll just store the documents
        # In a real implementation, you would also store embeddings
        self.documents.extend(docs)
        print(f"Added {len(docs)} documents to vector store")
    
    def query(self, text: str, top_k: int = 5) -> list:
        """Query the vector store for similar documents."""
        # For this example, we'll just return the first few documents
        # In a real implementation, you would compute similarity scores
        results = []
        for i, doc in enumerate(self.documents[:top_k]):
            result = {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": 1.0 / (i + 1)  # Simple decreasing scores
            }
            results.append(result)
        return results


class CustomLLM(BaseLLM):
    """A custom LLM that generates simple responses."""
    
    def generate(self, prompt: str, context: list = None) -> str:
        """Generate a response based on a prompt and optional context."""
        # For this example, we'll create a simple rule-based response
        # In a real implementation, you would use an actual LLM
        
        # Combine prompt and context
        full_text = prompt
        if context:
            context_texts = [doc["content"] for doc in context]
            full_text += " " + " ".join(context_texts)
        
        # Simple rule-based responses
        if "what is" in prompt.lower() and "nexusrag" in prompt.lower():
            return "NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data."
        elif "feature" in prompt.lower():
            return "Key features include multimodal parsing, modular design, and agent-ready capabilities."
        elif "capital" in prompt.lower() and "france" in prompt.lower():
            return "The capital of France is Paris."
        elif "created" in prompt.lower() and "python" in prompt.lower():
            return "The Python programming language was created by Guido van Rossum."
        else:
            # Default response
            return f"Based on the provided information, I can tell you that NexusRAG is a modular framework for building AI agents. Your query was: {prompt}"


def main():
    """Demonstrate usage of custom components with NexusRAG."""
    print("NexusRAG Custom Components Example")
    print("=" * 35)
    
    # Initialize custom components
    print("\nInitializing custom components...")
    parser = CustomParser()
    embedder = CustomEmbedder()
    vector_store = CustomVectorStore()
    llm = CustomLLM()
    
    # Initialize pipeline
    pipeline = RAGPipeline(parser, embedder, vector_store, llm)
    print("✓ Custom components initialized successfully")
    
    # Process a "document" (in this case, just a file path)
    print("\nProcessing document with custom parser...")
    sample_file_path = "sample_document.txt"
    pipeline.process_document(sample_file_path)
    print("✓ Document processed successfully")
    
    # Ask questions
    print("\nAsking questions with custom components...")
    questions = [
        "What is NexusRAG?",
        "What are the key features?",
        "What is the capital of France?",
        "Who created Python?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = pipeline.ask(question)
        print(f"A: {answer}")
    
    print("\n" + "=" * 35)
    print("Custom components example completed!")


if __name__ == "__main__":
    main()
