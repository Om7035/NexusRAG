#!/usr/bin/env python3
"""
Command-line interface for NexusRAG.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.enhanced_pipeline import EnhancedRAGPipeline
from nexusrag.parsers.universal import UniversalParser
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.llms.universal import UniversalLLM
from nexusrag.config.manager import ConfigManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NexusRAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process document(s)")
    process_parser.add_argument("files", nargs="+", help="Path(s) to the document file(s)")
    process_parser.add_argument("--no-chunk", action="store_true", help="Disable document chunking")
    process_parser.add_argument("--config", help="Path to configuration file")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about processed documents")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--filter-content-type", help="Filter by content type")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    ask_parser.add_argument("--config", help="Path to configuration file")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration information")
    config_parser.add_argument("--config", help="Path to configuration file")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--config", help="Path to configuration file")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the benchmark suite")
    benchmark_parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigManager(args.config if hasattr(args, 'config') and args.config else None)
    
    if args.command == "version":
        print(f"NexusRAG v0.1.0")
        return
    
    if args.command == "config":
        print("NexusRAG Configuration:")
        print(f"  Config file: {config_manager.config_path}")
        print(f"  Pipeline settings: {config_manager.get_pipeline_config()}")
        print(f"  Component settings: {config_manager.get_component_config('parser')}")
        print(f"  Advanced features: {config_manager.get_advanced_config()}")
        return
    
    if args.command == "process":
        # Initialize pipeline with config
        pipeline = create_pipeline(config_manager)
        
        # Process documents
        try:
            print(f"Processing {len(args.files)} document(s)...", file=sys.stderr)
            for file_path in args.files:
                print(f"  Processing {file_path}...", file=sys.stderr)
                pipeline.process_document(file_path, chunk=not args.no_chunk)
            print("Document(s) processed successfully!", file=sys.stderr)
        except Exception as e:
            print(f"Error processing document(s): {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.command == "ask":
        # Initialize pipeline with config
        pipeline = create_pipeline(config_manager)
        
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
    
    if args.command == "serve":
        # Import here to avoid dependency issues
        try:
            from nexusrag.api import create_app
            app = create_app()
            print(f"Starting NexusRAG API server on {args.host}:{args.port}")
            app.run(host=args.host, port=args.port, debug=False)
        except Exception as e:
            print(f"Error starting API server: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.command == "benchmark":
        # Import here to avoid dependency issues
        try:
            from .evaluation.benchmark import BenchmarkSuite
            benchmark = BenchmarkSuite()
            results = benchmark.run_full_benchmark()
            benchmark.print_report()
        except Exception as e:
            print(f"Error running benchmarks: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # If no command specified, show help
    parser.print_help()


def create_pipeline(config_manager: ConfigManager = None):
    """Create an enhanced RAG pipeline with configured components.
    
    Args:
        config_manager (ConfigManager): Configuration manager
        
    Returns:
        EnhancedRAGPipeline: Configured pipeline
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    # Get component configurations
    parser_config = config_manager.get_component_config("parser")
    embedder_config = config_manager.get_component_config("embedder")
    vector_store_config = config_manager.get_component_config("vector_store")
    llm_config = config_manager.get_component_config("llm")
    pipeline_config = config_manager.get_pipeline_config()
    
    # Initialize components
    parser = UniversalParser()
    embedder = UniversalEmbedder(
        provider=embedder_config.get("type", "sentence-transformers"),
        model_name=embedder_config.get("model")
    )
    vector_store = UniversalVectorStore(
        provider=vector_store_config.get("type", "chroma"),
        collection_name=vector_store_config.get("collection_name", "nexusrag")
    )
    llm = UniversalLLM(
        provider=llm_config.get("type", "huggingface"),
        model_name=llm_config.get("model")
    )
    
    # Create pipeline
    return EnhancedRAGPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        chunk_size=pipeline_config.get("chunk_size", 1000),
        chunk_overlap=pipeline_config.get("chunk_overlap", 200)
    )


if __name__ == "__main__":
    main()
