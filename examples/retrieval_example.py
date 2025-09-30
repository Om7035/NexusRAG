"""
Retrieval Engine Example for NexusRAG.

This example demonstrates the enhanced retrieval engine capabilities of NexusRAG.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.parsers.base import Document
from nexusrag.embedders.universal import UniversalEmbedder
from nexusrag.vectorstores.universal import UniversalVectorStore
from nexusrag.retrievers.universal import UniversalRetriever


def create_sample_documents():
    """Create sample documents for demonstration."""
    documents = [
        Document(
            content="Apple Inc. is a technology company founded by Steve Jobs.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="Microsoft is a technology company founded by Bill Gates.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="Google is a technology company founded by Larry Page and Sergey Brin.",
            metadata={"source": "tech_companies.txt", "content_type": "text"}
        ),
        Document(
            content="| Company | Founder | Current CEO |\n|---------|---------|-------------|\n| Apple Inc. | Steve Jobs | Tim Cook |\n| Microsoft | Bill Gates | Satya Nadella |\n| Google | Larry Page & Sergey Brin | Sundar Pichai |",
            metadata={"source": "company_table.txt", "content_type": "table"}
        ),
        Document(
            content="Steve Jobs was the CEO of Apple Inc. until his death in 2011.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Tim Cook is the current CEO of Apple Inc.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Bill Gates was the CEO of Microsoft until 2000.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
        Document(
            content="Satya Nadella is the current CEO of Microsoft.",
            metadata={"source": "ceos.txt", "content_type": "text"}
        ),
    ]
    
    return documents


def main():
    """Demonstrate retrieval engine capabilities."""
    print("NexusRAG Retrieval Engine Example")
    print("=" * 38)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Initialize components
    print("\nInitializing Retrieval Engine components...")
    embedder = UniversalEmbedder(provider="sentence-transformers")
    vector_store = UniversalVectorStore(provider="chroma")
    
    # Initialize universal retriever
    retriever = UniversalRetriever(vector_store, embedder, use_bge_reranker=False)
    print("✓ Components initialized successfully")
    
    # Add documents to retriever
    print("\nAdding documents to retrieval engine...")
    retriever.add_documents(documents)
    print("✓ Documents added successfully")
    
    # Perform standard search
    print("\nPerforming STANDARD search:")
    query = "Who is the current CEO of Apple?"
    print(f"Query: {query}")
    
    results = retriever.search(query, top_k=3)
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:50]}... (Score: {result.get('hybrid_score', result.get('score', 0.0)):.4f})")
    
    # Perform search without re-ranking
    print("\nPerforming search WITHOUT re-ranking:")
    query = "Who founded Microsoft?"
    print(f"Query: {query}")
    
    results = retriever.search(query, top_k=3, use_reranking=False)
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:50]}... (Score: {result.get('hybrid_score', result.get('score', 0.0)):.4f})")
    
    # Perform cross-modal search
    print("\nPerforming CROSS-MODAL search for tables:")
    query = "Show me company information"
    print(f"Query: {query}")
    
    results = retriever.cross_modal_search(query, modality="table", top_k=2)
    print(f"Found {len(results)} table results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:50]}... (Score: {result.get('score', 0.0):.4f})")
    
    # Perform multimodal fusion search
    print("\nPerforming MULTIMODAL FUSION search:")
    queries = {
        "text": "technology companies",
        "table": "company information"
    }
    print(f"Queries: {queries}")
    
    results = retriever.multimodal_fusion_search(queries, top_k=3)
    print(f"Found {len(results)} fused results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:50]}... (Score: {result.get('fused_score', 0.0):.4f})")
    
    # Get search metadata
    print("\nGetting search METADATA:")
    metadata = retriever.get_search_metadata(results)
    print(f"Metadata: {metadata}")
    
    print("\n" + "=" * 38)
    print("Retrieval engine example completed successfully!")
    print("\nNexusRAG now supports:")
    print("- Hybrid search (vector + keyword)")
    print("- Re-ranking for precision")
    print("- Cross-modal retrieval")
    print("- Multimodal fusion search")


if __name__ == "__main__":
    main()
