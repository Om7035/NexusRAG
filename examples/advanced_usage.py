"""
Advanced usage example for NexusRAG.

This example demonstrates how to use the advanced features of the NexusRAG framework.
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


def create_sample_documents(temp_dir):
    """Create sample documents for demonstration."""
    # Create a sample PDF
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title = Paragraph("NexusRAG Advanced Features Demo", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    content = [
        "This document demonstrates the advanced features of NexusRAG.",
        "NexusRAG supports multiple document types including PDF, Word, HTML, and Markdown.",
        "It also supports multiple embedding models like Sentence Transformers, OpenAI, and Cohere.",
        "For vector storage, you can use Chroma, Pinecone, or Weaviate.",
        "As for language models, NexusRAG supports Hugging Face, OpenAI, and Anthropic models.",
        "Additional features include document chunking, metadata filtering, and multimodal processing.",
        "Document chunking breaks large documents into smaller, manageable pieces.",
        "Metadata filtering allows you to search within specific document types or sources.",
        "Multimodal processing handles text, images, and tables within documents.",
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
    
    doc.build(story)
    
    # Create a sample Markdown file
    md_path = os.path.join(temp_dir, "sample.md")
    with open(md_path, 'w') as f:
        f.write("""# Sample Markdown Document

## Introduction

This is a sample Markdown document to demonstrate NexusRAG's capabilities.

## Features

- Document parsing
- Text embedding
- Vector storage
- Language modeling

## Conclusion

NexusRAG is a powerful framework for building AI applications.""")
    
    return [pdf_path, md_path]


def main():
    """Demonstrate advanced usage of NexusRAG."""
    print("NexusRAG Advanced Usage Example")
    print("=" * 35)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample documents
        document_paths = create_sample_documents(temp_dir)
        print(f"Created {len(document_paths)} sample documents")
        
        # Initialize components with different providers
        print("\nInitializing NexusRAG components...")
        parser = UniversalParser()
        embedder = UniversalEmbedder(provider="sentence-transformers")
        vector_store = UniversalVectorStore(provider="chroma")
        llm = UniversalLLM(provider="huggingface")
        
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
        
        # Process documents
        print("\nProcessing documents with chunking...")
        pipeline.process_documents(document_paths, chunk=True)
        print("✓ Documents processed successfully")
        
        # Ask questions
        print("\nAsking questions...")
        questions = [
            "What are the advanced features of NexusRAG?",
            "What document types does NexusRAG support?",
            "What embedding models are supported?",
            "What vector stores can be used?",
            "What language models are supported?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            answer = pipeline.ask(question)
            print(f"A: {answer}")
        
        # Ask with metadata filtering
        print("\nAsking questions with metadata filtering...")
        filter_criteria = {"content_type": "paragraph"}
        question = "What is the capital of France?"
        print(f"\nQ: {question} (filtered by content_type='paragraph')")
        answer = pipeline.ask(question, filter_metadata=filter_criteria)
        print(f"A: {answer}")
        
        print("\n" + "=" * 35)
        print("Advanced example completed successfully!")


if __name__ == "__main__":
    main()
