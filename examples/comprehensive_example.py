"""
Comprehensive example for NexusRAG.

This example demonstrates all the advanced features of the NexusRAG framework.
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
from nexusrag.chunking import DocumentChunker
from nexusrag.metadata_filter import MetadataFilter
from nexusrag.multimodal import MultimodalProcessor
from nexusrag.parsers.base import Document


def create_sample_documents(temp_dir):
    """Create various sample documents for demonstration."""
    documents = []
    
    # Create a sample PDF
    pdf_path = os.path.join(temp_dir, "sample.pdf")
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title = Paragraph("NexusRAG Comprehensive Demo", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    content = [
        "This document demonstrates all the features of NexusRAG.",
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
    
    for i, paragraph_text in enumerate(content):
        paragraph = Paragraph(paragraph_text, styles["Normal"])
        story.append(paragraph)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    documents.append(pdf_path)
    
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
    documents.append(md_path)
    
    # Create a sample text file
    txt_path = os.path.join(temp_dir, "sample.txt")
    with open(txt_path, 'w') as f:
        f.write("""Sample Text Document

This is a sample text document.

NexusRAG can process plain text files as well.

It extracts content and metadata from these files.

The quick brown fox jumps over the lazy dog.""")
    documents.append(txt_path)
    
    return documents


def demonstrate_chunking():
    """Demonstrate document chunking functionality."""
    print("\n1. Document Chunking Demo")
    print("-" * 30)
    
    # Create a long document
    long_content = "This is a very long document. " * 100
    doc = Document(content=long_content, metadata={"source": "long_doc.txt"})
    
    # Create chunker
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    
    # Chunk document
    chunks = chunker.chunk_document(doc)
    
    print(f"Original document length: {len(doc.content)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0].content)} characters")
    print(f"First chunk metadata: {chunks[0].metadata}")


def demonstrate_metadata_filtering():
    """Demonstrate metadata filtering functionality."""
    print("\n2. Metadata Filtering Demo")
    print("-" * 30)
    
    # Create sample documents with metadata
    docs = [
        Document("Content about AI", {"source": "ai_paper.pdf", "content_type": "paragraph", "topic": "AI"}),
        Document("Content about ML", {"source": "ml_paper.pdf", "content_type": "paragraph", "topic": "ML"}),
        Document("Table data", {"source": "data.xlsx", "content_type": "table", "topic": "data"}),
        Document("Image description", {"source": "diagram.png", "content_type": "image", "topic": "AI"})
    ]
    
    # Filter by content type
    paragraphs = MetadataFilter.filter_by_content_type(docs, "paragraph")
    print(f"Documents with content_type 'paragraph': {len(paragraphs)}")
    
    # Filter by source pattern
    ai_docs = MetadataFilter.filter_by_source(docs, "ai")
    print(f"Documents with 'ai' in source: {len(ai_docs)}")
    
    # Filter by custom field
    ai_topic_docs = MetadataFilter.filter_by_custom_field(docs, "topic", "AI")
    print(f"Documents with topic 'AI': {len(ai_topic_docs)}")
    
    # Filter with custom function
    custom_filtered = MetadataFilter.filter_by_metadata(
        docs,
        lambda meta: meta.get("topic") == "AI" and meta.get("content_type") == "paragraph"
    )
    print(f"AI paragraphs: {len(custom_filtered)}")


def demonstrate_multimodal_processing():
    """Demonstrate multimodal processing functionality."""
    print("\n3. Multimodal Processing Demo")
    print("-" * 30)
    
    # Note: For a full demo, we would need actual image files
    # This is just a conceptual demonstration
    print("MultimodalProcessor can handle:")
    print("- Text documents (PDF, Word, HTML, Markdown, Text)")
    print("- Images (with OCR to extract text)")
    print("- Tables (structured data)")
    print("\nTo process images, you need:")
    print("- Pillow library")
    print("- pytesseract library")
    print("- Tesseract OCR engine installed on your system")


def demonstrate_universal_components():
    """Demonstrate universal components functionality."""
    print("\n4. Universal Components Demo")
    print("-" * 30)
    
    print("UniversalParser supports:")
    print("- PDF files (.pdf)")
    print("- Word files (.docx)")
    print("- HTML files (.html, .htm)")
    print("- Markdown files (.md)")
    print("- Text files (.txt)")
    
    print("\nUniversalEmbedder supports:")
    print("- Sentence Transformers")
    print("- OpenAI")
    print("- Cohere")
    
    print("\nUniversalVectorStore supports:")
    print("- Chroma")
    print("- Pinecone")
    print("- Weaviate")
    
    print("\nUniversalLLM supports:")
    print("- Hugging Face")
    print("- OpenAI")
    print("- Anthropic")


def demonstrate_pipeline(temp_dir):
    """Demonstrate the enhanced pipeline with all features."""
    print("\n5. Enhanced Pipeline Demo")
    print("-" * 30)
    
    # Create sample documents
    document_paths = create_sample_documents(temp_dir)
    print(f"Created {len(document_paths)} sample documents")
    
    # Initialize components
    print("\nInitializing components...")
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
        chunk_size=300,
        chunk_overlap=50
    )
    print("✓ Components initialized successfully")
    
    # Process documents
    print("\nProcessing documents with chunking...")
    pipeline.process_documents(document_paths, chunk=True)
    print("✓ Documents processed successfully")
    
    # Ask questions
    print("\nAsking questions...")
    questions = [
        "What are the main features of NexusRAG?",
        "What document types does NexusRAG support?",
        "What embedding models are supported?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = pipeline.ask(question)
        print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
    
    # Ask with metadata filtering
    print("\nAsking questions with metadata filtering...")
    filter_criteria = {"content_type": "paragraph"}
    question = "What is the capital of France?"
    print(f"\nQ: {question} (filtered by content_type='paragraph')")
    answer = pipeline.ask(question, filter_metadata=filter_criteria)
    print(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")


def main():
    """Run comprehensive demonstration of NexusRAG features."""
    print("NexusRAG Comprehensive Example")
    print("=" * 40)
    
    # Demonstrate individual features
    demonstrate_chunking()
    demonstrate_metadata_filtering()
    demonstrate_multimodal_processing()
    demonstrate_universal_components()
    
    # Demonstrate pipeline with all features
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        demonstrate_pipeline(temp_dir)
    
    print("\n" + "=" * 40)
    print("Comprehensive example completed successfully!")
    print("\nThis example demonstrated:")
    print("- Document chunking")
    print("- Metadata filtering")
    print("- Multimodal processing")
    print("- Universal components")
    print("- Enhanced pipeline with all features")


if __name__ == "__main__":
    main()
