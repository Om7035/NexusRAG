#!/usr/bin/env python3
"""
Command-line interface for NexusRAG.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NexusRAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process document(s)")
    process_parser.add_argument("files", nargs="+", help="Path(s) to the document file(s)")
    process_parser.add_argument("--no-chunk", action="store_true", help="Disable document chunking")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about processed documents")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--filter-content-type", help="Filter by content type")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration information")
    
    args = parser.parse_args()
    
    if args.command == "version":
        from nexusrag import __version__
        print(f"NexusRAG v{__version__}")
        return
    
    if args.command == "config":
        print("NexusRAG Configuration:")
        print("  Parser: UniversalParser (supports PDF, Word, HTML, Markdown, Text)")
        print("  Embedder: UniversalEmbedder (supports Sentence Transformers, OpenAI, Cohere, Gemini)")
        print("  Vector Store: UniversalVectorStore (supports Chroma, Pinecone, Weaviate)")
        print("  LLM: UniversalLLM (supports Hugging Face, OpenAI, Anthropic, Gemini, Ollama)")
        print("  Advanced Features: Document chunking, metadata filtering, multimodal processing")
        return
    
    if args.command == "process":
        # Initialize pipeline
        pipeline = create_pipeline()
        
        # Process documents
        try:
            print(f"Processing {len(args.files)} document(s)...")
            for file_path in args.files:
                print(f"  Processing {file_path}...")
                pipeline.process_document(file_path, chunk=not args.no_chunk)
            print("Document(s) processed successfully!")
        except Exception as e:
            print(f"Error processing document(s): {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.command == "ask":
        # Initialize pipeline
        pipeline = create_pipeline()
        
        # Prepare metadata filter if provided
        filter_metadata = None
        if args.filter_content_type:
            filter_metadata = {"content_type": args.filter_content_type}
        
        # Ask question
        try:
            print(f"Question: {args.question}")
            answer = pipeline.ask(args.question, top_k=args.top_k, filter_metadata=filter_metadata)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error generating answer: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # If no command specified, show help
    parser.print_help()


def create_pipeline():
    """Create an enhanced RAG pipeline with default components."""
    parser = UniversalParser()
    embedder = UniversalEmbedder(provider="sentence-transformers")
    vector_store = UniversalVectorStore(provider="chroma")
    llm = UniversalLLM(provider="huggingface")
    return EnhancedRAGPipeline(parser, embedder, vector_store, llm, chunk_size=1000, chunk_overlap=200)


if __name__ == "__main__":
    main()
