"""
Ollama usage example for NexusRAG.

This example demonstrates how to use Ollama with NexusRAG for local LLM processing.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM


def create_sample_document(temp_dir):
    """Create a sample document for demonstration."""
    # Create a sample text file
    txt_path = os.path.join(temp_dir, "sample.txt")
    with open(txt_path, 'w') as f:
        f.write("""Sample Document for Ollama Demo

This document demonstrates how to use Ollama with NexusRAG.

Ollama is a powerful tool for running large language models locally.

Key benefits of using Ollama:
1. No internet required for processing
2. Complete privacy - data never leaves your machine
3. No API costs for inference
4. Full control over model selection and parameters

NexusRAG makes it easy to switch between different providers.

The capital of France is Paris.
Water boils at 100 degrees Celsius.
Python is a popular programming language.""")
    
    return txt_path


def main():
    """Demonstrate usage of Ollama with NexusRAG."""
    print("NexusRAG Ollama Example")
    print("=" * 30)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample document
        document_path = create_sample_document(temp_dir)
        print(f"Created sample document: {document_path}")
        
        # Initialize components with Ollama
        print("\nInitializing NexusRAG with Ollama...")
        
        # Note: You need to have Ollama installed and running
        # Download from: https://ollama.com/download
        # Run: ollama pull llama2
        
        try:
            parser = UniversalParser()
            embedder = UniversalEmbedder(provider="sentence-transformers")
            vector_store = UniversalVectorStore(provider="chroma")
            llm = UniversalLLM(provider="ollama", model_name="llama2")
            
            # Initialize enhanced pipeline
            pipeline = EnhancedRAGPipeline(
                parser=parser,
                embedder=embedder,
                vector_store=vector_store,
                llm=llm,
                chunk_size=500,
                chunk_overlap=100
            )
            print("✓ Components initialized successfully")
            
            # Process document
            print("\nProcessing document with local embedding...")
            pipeline.process_document(document_path, chunk=True)
            print("✓ Document processed successfully")
            
            # Ask questions
            print("\nAsking questions with local LLM...")
            questions = [
                "What are the benefits of using Ollama?",
                "What is the capital of France?",
                "What is Python?"
            ]
            
            for question in questions:
                print(f"\nQ: {question}")
                try:
                    answer = pipeline.ask(question)
                    print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
                except Exception as e:
                    print(f"Error generating answer: {e}")
                    print("Make sure you have Ollama installed and running.")
                    print("Download from: https://ollama.com/download")
                    print("Run: ollama pull llama2")
            
            print("\n" + "=" * 30)
            print("Ollama example completed successfully!")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            print("\nTo use Ollama, you need to:")
            print("1. Download and install Ollama from: https://ollama.com/download")
            print("2. Pull a model: ollama pull llama2")
            print("3. Ensure Ollama is running (it should start automatically after installation)")


if __name__ == "__main__":
    main()
