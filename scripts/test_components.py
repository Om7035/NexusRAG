#!/usr/bin/env python3
"""
Script to test individual components of the NexusRAG framework.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.pdf import PDFParser
from nexusrag.embedders.sentence_transformers import SentenceTransformerEmbedder
from nexusrag.vectorstores.chroma import ChromaVectorStore
from nexusrag.llms.huggingface import HuggingFaceLLM
from nexusrag.pipeline import RAGPipeline
from nexusrag.parsers.base import Document


def test_parser():
    """Test the PDF parser component."""
    print("Testing PDF Parser...")
    
    # Create a simple test PDF in memory
    try:
        parser = PDFParser()
        print("✓ PDFParser initialized successfully")
        
        # Note: We can't easily create a test PDF in memory, so we'll just test initialization
        # In a real scenario, you would test with an actual PDF file
        print("- Parser test completed (no PDF file provided)")
        
    except Exception as e:
        print(f"✗ Error testing parser: {e}")


def test_embedder():
    """Test the Sentence Transformer embedder component."""
    print("\nTesting Sentence Transformer Embedder...")
    
    try:
        embedder = SentenceTransformerEmbedder()
        print("✓ SentenceTransformerEmbedder initialized successfully")
        
        # Test embedding generation
        test_texts = [
            "This is a test sentence.",
            "This is another test sentence."
        ]
        embeddings = embedder.embed(test_texts)
        print(f"✓ Generated embeddings for {len(test_texts)} texts")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        
    except Exception as e:
        print(f"✗ Error testing embedder: {e}")


def test_vector_store():
    """Test the Chroma vector store component."""
    print("\nTesting Chroma Vector Store...")
    
    try:
        # Create a temporary directory for Chroma persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = ChromaVectorStore(persist_directory=temp_dir)
            print("✓ ChromaVectorStore initialized successfully")
            
            # Test adding documents
            test_docs = [
                Document("This is the first test document.", {"source": "test1.txt"}),
                Document("This is the second test document.", {"source": "test2.txt"})
            ]
            vector_store.add(test_docs)
            print("✓ Added 2 test documents to vector store")
            
            # Test querying
            results = vector_store.query("test document", top_k=1)
            print(f"✓ Queried vector store, found {len(results)} results")
            
    except Exception as e:
        print(f"✗ Error testing vector store: {e}")


def test_llm():
    """Test the Hugging Face LLM component."""
    print("\nTesting Hugging Face LLM...")
    
    try:
        llm = HuggingFaceLLM()
        print("✓ HuggingFaceLLM initialized successfully")
        
        # Test generation
        prompt = "What is the capital of France?"
        response = llm.generate(prompt)
        print(f"✓ Generated response to prompt: {response}")
        
    except Exception as e:
        print(f"✗ Error testing LLM: {e}")


def test_pipeline():
    """Test the RAG pipeline."""
    print("\nTesting RAG Pipeline...")
    
    try:
        # Initialize components
        parser = PDFParser()
        embedder = SentenceTransformerEmbedder()
        vector_store = ChromaVectorStore()
        llm = HuggingFaceLLM()
        
        # Initialize pipeline
        pipeline = RAGPipeline(parser, embedder, vector_store, llm)
        print("✓ RAGPipeline initialized successfully")
        
        # Note: We can't easily test the full pipeline without a PDF file
        # In a real scenario, you would test with an actual PDF file
        print("- Pipeline test completed (no PDF file provided)")
        
    except Exception as e:
        print(f"✗ Error testing pipeline: {e}")


def main():
    """Run all component tests."""
    print("NexusRAG Component Tests")
    print("=" * 30)
    
    test_parser()
    test_embedder()
    test_vector_store()
    test_llm()
    test_pipeline()
    
    print("\n" + "=" * 30)
    print("Component tests completed!")


if __name__ == "__main__":
    main()
