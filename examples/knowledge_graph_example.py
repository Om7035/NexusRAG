"""
Knowledge graph example for NexusRAG.

This example demonstrates how to use knowledge graph features with NexusRAG.
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
    documents = []
    
    # Create a sample text file with entities and relationships
    txt_path = os.path.join(temp_dir, "sample_kg.txt")
    with open(txt_path, 'w') as f:
        f.write("""Knowledge Graph Demo Document

This document contains information about companies and their employees.

Apple Inc. is a technology company founded by Steve Jobs.
Steve Jobs was the CEO of Apple Inc. until his death in 2011.
Tim Cook is the current CEO of Apple Inc.

Microsoft is a technology company founded by Bill Gates.
Bill Gates was the CEO of Microsoft until 2000.
Satya Nadella is the current CEO of Microsoft.

Google is a technology company founded by Larry Page and Sergey Brin.
Larry Page and Sergey Brin were the CEOs of Google until 2015.
Sundar Pichai is the current CEO of Google.

All these companies are based in the United States.
They are all part of the technology industry.
""")
    documents.append(txt_path)
    
    # Create another sample document
    txt_path2 = os.path.join(temp_dir, "sample_kg2.txt")
    with open(txt_path2, 'w') as f:
        f.write("""Additional Information

Apple Inc. was founded in 1976 in Cupertino, California.
Microsoft was founded in 1975 in Albuquerque, New Mexico.
Google was founded in 1998 in Menlo Park, California.

Steve Jobs was born in 1955 and died in 2011.
Bill Gates was born in 1955.
Tim Cook was born in 1960.

The technology industry includes companies that develop software and hardware products.
""")
    documents.append(txt_path2)
    
    return documents


def main():
    """Demonstrate knowledge graph features of NexusRAG."""
    print("NexusRAG Knowledge Graph Example")
    print("=" * 35)
    
    # Create a temporary directory for our files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample documents
        document_paths = create_sample_documents(temp_dir)
        print(f"Created {len(document_paths)} sample documents")
        
        # Initialize components
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
        print("\nProcessing documents and building knowledge graph...")
        pipeline.process_documents(document_paths, chunk=True)
        print("✓ Documents processed and knowledge graph built successfully")
        
        # Show knowledge graph information
        if pipeline.knowledge_graph:
            print(f"\nKnowledge Graph Information:")
            print(f"  Entities: {len(pipeline.knowledge_graph.entities)}")
            print(f"  Relationships: {len(pipeline.knowledge_graph.relationships)}")
        
        # Ask questions with and without knowledge graph
        print("\nAsking questions...")
        questions = [
            "Who is the current CEO of Apple?",
            "When was Microsoft founded?",
            "Who founded Google?",
            "What industry are Apple, Microsoft, and Google part of?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            
            # Ask without knowledge graph
            answer1 = pipeline.ask(question)
            print(f"A (without KG): {answer1[:100]}..." if len(answer1) > 100 else f"A (without KG): {answer1}")
            
            # Ask with knowledge graph (placeholder for now)
            answer2 = pipeline.ask(question, use_knowledge_graph=True)
            print(f"A (with KG): {answer2[:100]}..." if len(answer2) > 100 else f"A (with KG): {answer2}")
        
        print("\n" + "=" * 35)
        print("Knowledge graph example completed successfully!")
        print("\nNote: Knowledge graph features are a placeholder in this example.")
        print("In a full implementation, they would provide enhanced reasoning capabilities.")


if __name__ == "__main__":
    main()
