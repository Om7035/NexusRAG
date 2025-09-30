"""
Basic usage example for NexusRAG.

This example demonstrates how to use the NexusRAG framework to process a document
and ask questions about its content.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.pipeline import RAGPipeline
from nexusrag.parsers.pdf import PDFParser
from nexusrag.embedders.sentence_transformers import SentenceTransformerEmbedder
from nexusrag.vectorstores.chroma import ChromaVectorStore
from nexusrag.llms.huggingface import HuggingFaceLLM


def create_sample_pdf(filename):
    """Create a sample PDF document for demonstration."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("NexusRAG Demonstration Document", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add content sections
    sections = [
        ("Introduction", [
            "NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data.",
            "It combines high-fidelity document parsing with a fully modular architecture.",
            "The framework enables developers to create powerful, data-aware applications."
        ]),
        ("Key Features", [
            "Multimodal Parsing: High-fidelity extraction from PDFs, documents, and other formats.",
            "Modular Design: Pluggable components for parsers, embedders, vector stores, and LLMs.",
            "Agent-Ready: Built for creating autonomous AI agents that can reason over complex data.",
            "Extensible: Easy to add new components and customize existing ones."
        ]),
        ("Technical Details", [
            "NexusRAG follows a modular architecture with four main components: Parsers, Embedders, Vector Stores, and LLMs.",
            "These components are orchestrated by the Pipeline, which provides a unified interface for processing documents and answering questions.",
            "The framework is built with Python and leverages state-of-the-art libraries for each component."
        ]),
        ("Example Facts", [
            "The capital of France is Paris.",
            "The largest planet in our solar system is Jupiter.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The Python programming language was created by Guido van Rossum.",
            "Machine learning is a subset of artificial intelligence."
        ])
    ]
    
    for section_title, section_content in sections:
        # Add section title
        section_header = Paragraph(section_title, styles["Heading2"])
        story.append(section_header)
        story.append(Spacer(1, 12))
        
        # Add section content
        for paragraph_text in section_content:
            paragraph = Paragraph(paragraph_text, styles["Normal"])
            story.append(paragraph)
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    return filename


def main():
    """Demonstrate basic usage of NexusRAG."""
    print("NexusRAG Basic Usage Example")
    print("=" * 30)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample PDF
        sample_pdf_path = os.path.join(temp_dir, "sample_document.pdf")
        create_sample_pdf(sample_pdf_path)
        print(f"Created sample PDF: {sample_pdf_path}")
        
        # Initialize components
        print("\nInitializing NexusRAG components...")
        parser = PDFParser()
        embedder = SentenceTransformerEmbedder()
        vector_store = ChromaVectorStore(persist_directory=os.path.join(temp_dir, "chroma_db"))
        llm = HuggingFaceLLM()
        
        # Initialize pipeline
        pipeline = RAGPipeline(parser, embedder, vector_store, llm)
        print("✓ Components initialized successfully")
        
        # Process document
        print("\nProcessing document...")
        pipeline.process_document(sample_pdf_path)
        print("✓ Document processed successfully")
        
        # Ask questions
        print("\nAsking questions...")
        questions = [
            "What is NexusRAG?",
            "What are the key features of NexusRAG?",
            "What is the capital of France?",
            "Who created Python?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            answer = pipeline.ask(question)
            print(f"A: {answer}")
        
        print("\n" + "=" * 30)
        print("Example completed successfully!")


if __name__ == "__main__":
    main()
