#!/usr/bin/env python3
"""
Benchmark script for NexusRAG components.
"""

import time
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.pdf import PDFParser
from nexusrag.embedders.sentence_transformers import SentenceTransformerEmbedder
from nexusrag.vectorstores.chroma import ChromaVectorStore
from nexusrag.llms.huggingface import HuggingFaceLLM
from nexusrag.pipeline import RAGPipeline


def create_test_pdf(filename):
    """Create a simple test PDF with known content."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    # Create document
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("Benchmark Test Document for NexusRAG", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add some content
    content = [
        "This is a test document created for benchmarking NexusRAG components.",
        "NexusRAG is an open-source framework for building autonomous AI agents that reason over complex, multimodal data.",
        "It combines high-fidelity document parsing with a fully modular architecture.",
        "The framework enables developers to create powerful, data-aware applications.",
        "Key features include multimodal parsing, modular design, and agent-ready capabilities.",
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Python programming language was created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence."
    ] * 10  # Repeat content to make it larger
    
    for paragraph_text in content:
        paragraph = Paragraph(paragraph_text, styles["Normal"])
        story.append(paragraph)
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    return filename


def benchmark_parser():
    """Benchmark the PDF parser component."""
    print("Benchmarking PDF Parser...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test PDF
        test_pdf_path = os.path.join(temp_dir, "benchmark_document.pdf")
        create_test_pdf(test_pdf_path)
        
        # Initialize parser
        parser = PDFParser()
        
        # Benchmark parsing
        start_time = time.time()
        documents = parser.parse(test_pdf_path)
        end_time = time.time()
        
        print(f"  Parsed {len(documents)} documents in {end_time - start_time:.2f} seconds")
        return end_time - start_time


def benchmark_embedder():
    """Benchmark the Sentence Transformer embedder component."""
    print("Benchmarking Sentence Transformer Embedder...")
    
    # Initialize embedder
    embedder = SentenceTransformerEmbedder()
    
    # Create test texts
    test_texts = [
        "This is a test sentence for benchmarking.",
        "This is another test sentence for benchmarking.",
        "NexusRAG is an open-source framework for building autonomous AI agents.",
        "It combines high-fidelity document parsing with a fully modular architecture.",
        "The framework enables developers to create powerful, data-aware applications."
    ] * 20  # Repeat to make it larger
    
    # Benchmark embedding
    start_time = time.time()
    embeddings = embedder.embed(test_texts)
    end_time = time.time()
    
    print(f"  Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    return end_time - start_time


def benchmark_vector_store():
    """Benchmark the Chroma vector store component."""
    print("Benchmarking Chroma Vector Store...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize vector store
        vector_store = ChromaVectorStore(persist_directory=os.path.join(temp_dir, "chroma_db"))
        
        # Create test documents
        from nexusrag.parsers.base import Document
        test_docs = [
            Document(f"This is test document {i} for benchmarking.", {"source": f"test{i}.txt"})
            for i in range(100)
        ]
        
        # Benchmark adding documents
        start_time = time.time()
        vector_store.add(test_docs)
        add_time = time.time() - start_time
        
        # Benchmark querying
        start_time = time.time()
        results = vector_store.query("test document", top_k=5)
        query_time = time.time() - start_time
        
        print(f"  Added {len(test_docs)} documents in {add_time:.2f} seconds")
        print(f"  Queried vector store in {query_time:.2f} seconds")
        return add_time + query_time


def benchmark_llm():
    """Benchmark the Hugging Face LLM component."""
    print("Benchmarking Hugging Face LLM...")
    
    # Initialize LLM
    llm = HuggingFaceLLM()
    
    # Benchmark generation
    prompt = "Explain what artificial intelligence is in one sentence."
    start_time = time.time()
    response = llm.generate(prompt)
    end_time = time.time()
    
    print(f"  Generated response in {end_time - start_time:.2f} seconds")
    return end_time - start_time


def benchmark_pipeline():
    """Benchmark the full RAG pipeline."""
    print("Benchmarking Full RAG Pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test PDF
        test_pdf_path = os.path.join(temp_dir, "benchmark_document.pdf")
        create_test_pdf(test_pdf_path)
        
        # Initialize components
        parser = PDFParser()
        embedder = SentenceTransformerEmbedder()
        vector_store = ChromaVectorStore(persist_directory=os.path.join(temp_dir, "chroma_db"))
        llm = HuggingFaceLLM()
        
        # Initialize pipeline
        pipeline = RAGPipeline(parser, embedder, vector_store, llm)
        
        # Benchmark document processing
        start_time = time.time()
        pipeline.process_document(test_pdf_path)
        process_time = time.time() - start_time
        
        # Benchmark question answering
        start_time = time.time()
        answer = pipeline.ask("What is NexusRAG?")
        ask_time = time.time() - start_time
        
        print(f"  Processed document in {process_time:.2f} seconds")
        print(f"  Generated answer in {ask_time:.2f} seconds")
        return process_time + ask_time


def main():
    """Run all benchmarks."""
    print("NexusRAG Benchmark Suite")
    print("=" * 30)
    
    # Try to import reportlab for PDF creation
    try:
        import reportlab
    except ImportError:
        print("ReportLab library not found. Please install it with: pip install reportlab")
        print("Skipping benchmarks.")
        return
    
    # Run benchmarks
    parser_time = benchmark_parser()
    embedder_time = benchmark_embedder()
    vector_store_time = benchmark_vector_store()
    llm_time = benchmark_llm()
    pipeline_time = benchmark_pipeline()
    
    # Print summary
    print("\n" + "=" * 30)
    print("Benchmark Summary:")
    print(f"  PDF Parser:        {parser_time:.2f} seconds")
    print(f"  Text Embedder:     {embedder_time:.2f} seconds")
    print(f"  Vector Store:      {vector_store_time:.2f} seconds")
    print(f"  Language Model:    {llm_time:.2f} seconds")
    print(f"  Full Pipeline:     {pipeline_time:.2f} seconds")
    print("\nNote: Times may vary based on system performance and model loading.")


if __name__ == "__main__":
    main()
