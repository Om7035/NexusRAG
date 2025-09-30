"""
Quickstart example for NexusRAG.

This example demonstrates the key features of NexusRAG in a simple, easy-to-understand way.
"""

import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag import RAG


def create_sample_documents():
    """Create sample documents for demonstration."""
    # Create a temporary directory for our files
    temp_dir = tempfile.mkdtemp()
    
    # Create a sample text file
    txt_path = os.path.join(temp_dir, "sample.txt")
    with open(txt_path, 'w') as f:
        f.write("""NexusRAG Quickstart Guide
        
        Welcome to NexusRAG! This is an open-source framework for building 
        autonomous AI agents that reason over complex, multimodal data.
        
        Key Features:
        - Multimodal Processing: Text, PDF, images, audio, video, tables
        - Smart Chunking: Character, semantic, and sentence-based strategies
        - Hybrid Search: Vector + keyword search with re-ranking
        - Multi-Step Reasoning: Iterative refinement with citations
        - Local LLM Support: Full Ollama integration
        
        The capital of France is Paris.
        The largest planet in our solar system is Jupiter.
        Water boils at 100 degrees Celsius at sea level.
        """)
    
    return [txt_path]


def main():
    """Demonstrate NexusRAG quickstart."""
    print("NexusRAG Quickstart Example")
    print("=" * 30)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"✓ Created {len(documents)} sample document(s)")
    
    # Initialize RAG with default settings
    print("\nInitializing NexusRAG...")
    rag = RAG()
    print("✓ NexusRAG initialized successfully")
    
    # Process documents
    print("\nProcessing documents...")
    rag.process(documents)
    print("✓ Documents processed successfully")
    
    # Ask simple questions
    print("\nAsking questions...")
    
    questions = [
        "What is NexusRAG?",
        "What are the key features of NexusRAG?",
        "What is the capital of France?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.ask(question)
        print(f"A: {answer[:100]}..." if len(answer) > 100 else f"A: {answer}")
    
    # Ask with reasoning
    print("\nAsking with reasoning...")
    reasoning_question = "Explain why NexusRAG is useful for developers?"
    print(f"Q: {reasoning_question}")
    reasoning_answer = rag.ask_with_reasoning(reasoning_question, max_steps=3)
    print(f"A: {reasoning_answer[:200]}..." if len(reasoning_answer) > 200 else f"A: {reasoning_answer}")
    
    # Clean up
    for doc in documents:
        os.unlink(doc)
    os.rmdir(os.path.dirname(documents[0]))
    
    print("\n" + "=" * 30)
    print("Quickstart example completed successfully!")
    print("\nNexusRAG is ready for production use! 🚀")


if __name__ == "__main__":
    main()
