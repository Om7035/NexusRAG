"""
Enhanced knowledge graph example for NexusRAG.

This example demonstrates the enhanced knowledge graph features of NexusRAG.
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
from nexusrag.knowledge_graph import KnowledgeGraphBuilder


def create_sample_documents(temp_dir):
    """Create sample documents for demonstration."""
    documents = []
    
    # Create a sample text file with entities and relationships
    txt_path = os.path.join(temp_dir, "sample_kg.txt")
    with open(txt_path, 'w') as f:
        f.write("""Knowledge Graph Demo Document

This document contains information about companies and their founders.

Steve Jobs founded Apple Inc. in 1976 in Cupertino, California.
Apple Inc. is a technology company that develops consumer electronics.

Bill Gates founded Microsoft Corporation in 1975 in Albuquerque, New Mexico.
Microsoft Corporation is a technology company that develops software products.

Larry Page and Sergey Brin founded Google Inc. in 1998 in Menlo Park, California.
Google Inc. is a technology company that provides internet-related services.

All these companies are based in the United States.
They are all part of the technology industry.
""")
    documents.append(txt_path)
    
    # Create another sample document
    txt_path2 = os.path.join(temp_dir, "sample_kg2.txt")
    with open(txt_path2, 'w') as f:
        f.write("""Additional Information

Apple Inc. was founded in 1976 and is headquartered in Cupertino.
Microsoft was founded in 1975 and is headquartered in Redmond, Washington.
Google was founded in 1998 and is headquartered in Mountain View, California.

Steve Jobs was born in 1955 and died in 2011.
Bill Gates was born in 1955.
Larry Page was born in 1973.
Sergey Brin was born in 1973.

The technology industry includes companies that develop software and hardware products.
""")
    documents.append(txt_path2)
    
    return documents


def main():
    """Demonstrate enhanced knowledge graph features of NexusRAG."""
    print("NexusRAG Enhanced Knowledge Graph Example")
    print("=" * 45)
    
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
            
            # Show entity types
            entity_types = {}
            for entity in pipeline.knowledge_graph.entities.values():
                entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
            
            print(f"  Entity Types:")
            for entity_type, count in entity_types.items():
                print(f"    {entity_type}: {count}")
            
            # Search for specific entities
            print(f"\nSearching for entities:")
            apple_entities = pipeline.knowledge_graph.search_entities("Apple")
            print(f"  Apple entities found: {len(apple_entities)}")
            
            steve_entities = pipeline.knowledge_graph.search_entities("Steve Jobs")
            print(f"  Steve Jobs entities found: {len(steve_entities)}")
        
        # Ask questions with and without knowledge graph
        print("\nAsking questions...")
        questions = [
            "Who founded Apple Inc.?",
            "When was Microsoft founded?",
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
        
        print("\n" + "=" * 45)
        print("Enhanced knowledge graph example completed successfully!")
        print("\nNote: Knowledge graph features are now functional with entity extraction.")
        print("Future versions will include enhanced reasoning capabilities.")


if __name__ == "__main__":
    main()
