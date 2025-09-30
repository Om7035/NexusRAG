"""
Free API usage example for NexusRAG.

This example demonstrates how to use free APIs like Google Gemini with NexusRAG.
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
        f.write("""Sample Document for Free API Demo

This document demonstrates how to use free APIs with NexusRAG.

Google Gemini is a free alternative to paid APIs like OpenAI and Anthropic.

Key benefits of using free APIs:
1. No cost for experimentation and development
2. Good performance for many use cases
3. Easy to get started without payment information
4. Generous free tiers for initial usage

NexusRAG makes it easy to switch between different providers.

The capital of France is Paris.
Water boils at 100 degrees Celsius.
Python is a popular programming language.""")
    
    return txt_path


def main():
    """Demonstrate usage of free APIs with NexusRAG."""
    print("NexusRAG Free API Example")
    print("=" * 30)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample document
        document_path = create_sample_document(temp_dir)
        print(f"Created sample document: {document_path}")
        
        # Initialize components with free APIs (Google Gemini)
        print("\nInitializing NexusRAG with free APIs (Google Gemini)...")
        
        # Note: You need to set your GEMINI_API_KEY environment variable
        # export GEMINI_API_KEY="your-google-ai-key"
        
        try:
            parser = UniversalParser()
            embedder = UniversalEmbedder(provider="gemini")
            vector_store = UniversalVectorStore(provider="chroma")
            llm = UniversalLLM(provider="gemini")
            
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
            print("\nProcessing document with Gemini embedding...")
            pipeline.process_document(document_path, chunk=True)
            print("✓ Document processed successfully")
            
            # Ask questions
            print("\nAsking questions with Gemini LLM...")
            questions = [
                "What are the benefits of using free APIs?",
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
                    print("Make sure you have set your GEMINI_API_KEY environment variable.")
            
            print("\n" + "=" * 30)
            print("Free API example completed successfully!")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            print("\nTo use Google Gemini, you need to:")
            print("1. Get a free API key from Google AI Studio: https://aistudio.google.com/")
            print("2. Set the GEMINI_API_KEY environment variable")
            print("   export GEMINI_API_KEY='your-api-key'")


if __name__ == "__main__":
    main()
