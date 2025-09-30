import pytest
from nexusrag.knowledge_graph import Entity, Relationship, KnowledgeGraph, KnowledgeGraphBuilder
from nexusrag.parsers.base import Document


def test_entity_creation():
    """Test creating an entity."""
    entity = Entity("1", "Apple Inc.", "company", {"founded": 1976})
    assert entity.id == "1"
    assert entity.name == "Apple Inc."
    assert entity.type == "company"
    assert entity.properties["founded"] == 1976


def test_relationship_creation():
    """Test creating a relationship."""
    relationship = Relationship("1", "2", "founded_by", {"year": 1976})
    assert relationship.source_id == "1"
    assert relationship.target_id == "2"
    assert relationship.type == "founded_by"
    assert relationship.properties["year"] == 1976


def test_knowledge_graph():
    """Test knowledge graph functionality."""
    graph = KnowledgeGraph()
    
    # Add entities
    apple = Entity("1", "Apple Inc.", "company")
    steve = Entity("2", "Steve Jobs", "person")
    graph.add_entity(apple)
    graph.add_entity(steve)
    
    # Add relationship
    founded = Relationship("2", "1", "founded")
    graph.add_relationship(founded)
    
    # Test retrieval
    assert graph.get_entity("1").name == "Apple Inc."
    assert len(graph.get_entities_by_type("company")) == 1
    assert len(graph.get_relationships_by_type("founded")) == 1
    assert len(graph.get_relationships_for_entity("1")) == 1
    
    # Test search functionality
    results = graph.search_entities("Apple")
    assert len(results) == 1
    assert results[0].name == "Apple Inc."


def test_knowledge_graph_builder():
    """Test knowledge graph builder."""
    builder = KnowledgeGraphBuilder()
    
    # Create sample documents with entities and relationships
    doc1 = Document("Steve Jobs founded Apple Inc. in 1976.", {"source": "test1.txt"})
    doc2 = Document("Bill Gates founded Microsoft Corporation in 1975.", {"source": "test2.txt"})
    documents = [doc1, doc2]
    
    # Build knowledge graph
    graph = builder.build_from_documents(documents)
    
    # Test that graph was created
    assert isinstance(graph, KnowledgeGraph)
    # Should have document entities plus extracted entities
    assert len(graph.entities) >= 2
    # Should have relationships between entities and documents
    assert len(graph.relationships) >= 2
    
    # Test entity search
    apple_entities = graph.search_entities("Apple")
    assert len(apple_entities) >= 1
    
    steve_entities = graph.search_entities("Steve Jobs")
    assert len(steve_entities) >= 1
