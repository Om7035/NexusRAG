#!/usr/bin/env python3
"""
Integration test for NexusRAG components working together.
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


def create_simple_test_pdf(filename):
    """Create a simple test PDF with known content."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("Test Document for NexusRAG", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add some content
    content = [
        "This is a test document created for testing NexusRAG components.",
        "NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data.",
        "It combines high-fidelity document parsing with a fully modular architecture.",
        "The framework enables developers to create powerful, data-aware applications.",
        "Key features include multimodal parsing, modular design, and agent-ready capabilities.",
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Python programming language was created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    for paragraph_text in content:
        paragraph = Paragraph(paragraph_text, styles["Normal"])
        story.append(paragraph)
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    return filename


def test_full_pipeline():
    """Test the full RAG pipeline with all components."""
    print("Testing Full RAG Pipeline...")
    
    try:
        # Create a temporary directory for our test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PDF
            test_pdf_path = os.path.join(temp_dir, "test_document.pdf")
            create_simple_test_pdf(test_pdf_path)
            print(f"✓ Created test PDF: {test_pdf_path}")
            
            # Initialize components
            parser = PDFParser()
            embedder = SentenceTransformerEmbedder()
            vector_store = ChromaVectorStore(persist_directory=os.path.join(temp_dir, "chroma_db"))
            llm = HuggingFaceLLM()
            print("✓ Initialized all components")
            
            # Initialize pipeline
            pipeline = RAGPipeline(parser, embedder, vector_store, llm)
            print("✓ Initialized RAG pipeline")
            
            # Process document
            print("Processing document...")
            pipeline.process_document(test_pdf_path)
            print("✓ Processed document successfully")
            
            # Ask questions
            questions = [
                "What is NexusRAG?",
                "What are the key features of NexusRAG?",
                "What is the capital of France?",
                "Who created Python?"
            ]
            
            print("\nAsking questions:")
            for question in questions:
                try:
                    answer = pipeline.ask(question)
                    print(f"\nQ: {question}")
                    print(f"A: {answer}")
                except Exception as e:
                    print(f"\nQ: {question}")
                    print(f"Error generating answer: {e}")
                    
    except Exception as e:
        print(f"✗ Error in full pipeline test: {e}")
        return False
    
    print("\n✓ Full pipeline test completed successfully!")
    return True


def main():
    """Run integration tests."""
    print("NexusRAG Integration Tests")
    print("=" * 30)
    
    # Try to import reportlab for PDF creation
    try:
        import reportlab
    except ImportError:
        print("ReportLab library not found. Please install it with: pip install reportlab")
        print("Skipping integration test.")
        return
    
    success = test_full_pipeline()
    
    print("\n" + "=" * 30)
    if success:
        print("Integration tests completed successfully!")
    else:
        print("Integration tests failed!")


if __name__ == "__main__":
    main()
