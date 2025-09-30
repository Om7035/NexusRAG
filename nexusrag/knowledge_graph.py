import re
import hashlib
from typing import List, Dict, Any, Set
from collections import defaultdict
from .parsers.base import Document


class Entity:
    """Represents an entity in the knowledge graph."""
    
    def __init__(self, id: str, name: str, type: str, properties: Dict[str, Any] = None):
        """Initialize an entity.
        
        Args:
            id (str): Unique identifier for the entity
            name (str): Name of the entity
            type (str): Type of the entity (e.g., person, organization, concept)
            properties (Dict[str, Any]): Additional properties of the entity
        """
        self.id = id
        self.name = name
        self.type = type
        self.properties = properties or {}


class Relationship:
    """Represents a relationship between two entities."""
    
    def __init__(self, source_id: str, target_id: str, type: str, properties: Dict[str, Any] = None):
        """Initialize a relationship.
        
        Args:
            source_id (str): ID of the source entity
            target_id (str): ID of the target entity
            type (str): Type of relationship (e.g., works_for, located_in)
            properties (Dict[str, Any]): Additional properties of the relationship
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.properties = properties or {}


class KnowledgeGraph:
    """Represents a knowledge graph with entities and relationships."""
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph.
        
        Args:
            entity (Entity): Entity to add
        """
        self.entities[entity.id] = entity
        # Index entity name for quick lookup
        self.entity_index[entity.name.lower()].add(entity.id)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the knowledge graph.
        
        Args:
            relationship (Relationship): Relationship to add
        """
        self.relationships.append(relationship)
    
    def get_entity(self, id: str) -> Entity:
        """Get an entity by ID.
        
        Args:
            id (str): ID of the entity to retrieve
            
        Returns:
            Entity: The entity with the given ID, or None if not found
        """
        return self.entities.get(id)
    
    def get_entities_by_type(self, type: str) -> List[Entity]:
        """Get all entities of a specific type.
        
        Args:
            type (str): Type of entities to retrieve
            
        Returns:
            List[Entity]: List of entities of the specified type
        """
        return [entity for entity in self.entities.values() if entity.type == type]
    
    def get_relationships_by_type(self, type: str) -> List[Relationship]:
        """Get all relationships of a specific type.
        
        Args:
            type (str): Type of relationships to retrieve
            
        Returns:
            List[Relationship]: List of relationships of the specified type
        """
        return [rel for rel in self.relationships if rel.type == type]
    
    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for a specific entity.
        
        Args:
            entity_id (str): ID of the entity
            
        Returns:
            List[Relationship]: List of relationships involving the entity
        """
        return [rel for rel in self.relationships if rel.source_id == entity_id or rel.target_id == entity_id]
    
    def search_entities(self, query: str) -> List[Entity]:
        """Search for entities by name.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Entity]: List of matching entities
        """
        query = query.lower()
        matching_ids = set()
        
        # Exact match
        if query in self.entity_index:
            matching_ids.update(self.entity_index[query])
        
        # Partial match
        for name, ids in self.entity_index.items():
            if query in name:
                matching_ids.update(ids)
        
        return [self.entities[entity_id] for entity_id in matching_ids if entity_id in self.entities]


class KnowledgeGraphBuilder:
    """Builds knowledge graphs from documents with entity extraction."""
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        pass
    
    def build_from_documents(self, documents: List[Document]) -> KnowledgeGraph:
        """Build a knowledge graph from a list of documents.
        
        Args:
            documents (List[Document]): Documents to build the knowledge graph from
            
        Returns:
            KnowledgeGraph: Built knowledge graph
        """
        graph = KnowledgeGraph()
        
        # Process each document
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            
            # Add document entity
            doc_entity = Entity(
                id=doc_id,
                name=f"Document {i}",
                type="document",
                properties={
                    "content": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_type": doc.metadata.get("content_type", "unknown")
                }
            )
            graph.add_entity(doc_entity)
            
            # Extract entities and relationships from document content
            entities = self._extract_entities(doc.content)
            relationships = self._extract_relationships(doc.content, entities)
            
            # Add extracted entities to graph
            for entity in entities:
                graph.add_entity(entity)
                
                # Link entity to document
                doc_rel = Relationship(
                    source_id=entity.id,
                    target_id=doc_id,
                    type="mentioned_in"
                )
                graph.add_relationship(doc_rel)
            
            # Add extracted relationships to graph
            for relationship in relationships:
                graph.add_relationship(relationship)
        
        return graph
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using basic patterns.
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            List[Entity]: List of extracted entities
        """
        entities = []
        
        # Person names (simplified pattern)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        persons = re.findall(person_pattern, text)
        for person in persons:
            entity_id = f"person_{hashlib.md5(person.encode()).hexdigest()[:8]}"
            entity = Entity(
                id=entity_id,
                name=person,
                type="person"
            )
            entities.append(entity)
        
        # Organizations (simplified pattern)
        org_patterns = [
            r'\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd)\.?'
        ]
        
        for pattern in org_patterns:
            orgs = re.findall(pattern, text)
            for org in orgs:
                # re.findall returns tuples for groups, so we need to handle that
                if isinstance(org, tuple):
                    org_name = org[0]  # Get the full match
                else:
                    org_name = org
                    
                entity_id = f"org_{hashlib.md5(org_name.encode()).hexdigest()[:8]}"
                entity = Entity(
                    id=entity_id,
                    name=org_name,
                    type="organization"
                )
                entities.append(entity)
        
        # Technologies/concepts (simplified pattern)
        tech_pattern = r'\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)\b'
        technologies = re.findall(tech_pattern, text)
        
        # Filter out common words and short terms
        filtered_techs = [tech for tech in technologies 
                         if len(tech) > 3 and 
                         tech.lower() not in ['The', 'This', 'That', 'With', 'From', 'Into', 'Over', 'Under']]
        
        for tech in filtered_techs[:10]:  # Limit to avoid too many entities
            entity_id = f"tech_{hashlib.md5(tech.encode()).hexdigest()[:8]}"
            entity = Entity(
                id=entity_id,
                name=tech,
                type="technology"
            )
            entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities.
        
        Args:
            text (str): Text to extract relationships from
            entities (List[Entity]): Entities to find relationships between
            
        Returns:
            List[Relationship]: List of extracted relationships
        """
        relationships = []
        
        # Create a mapping of entity names to IDs
        entity_map = {entity.name.lower(): entity.id for entity in entities}
        
        # Simple relationship patterns
        patterns = [
            (r'([A-Z][a-z]+ [A-Z][a-z]+) is the (\w+) of ([A-Z][a-zA-Z]+ Inc)', 'is_role_of'),
            (r'([A-Z][a-z]+ [A-Z][a-z]+) founded ([A-Z][a-zA-Z]+ Inc)', 'founded'),
            (r'([A-Z][a-z]+ [A-Z][a-z]+) works at ([A-Z][a-zA-Z]+ Inc)', 'works_at'),
            (r'([A-Z][a-z]+ [A-Z][a-z]+) is (\w+) at ([A-Z][a-zA-Z]+ Inc)', 'is_role_at')
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source_name = groups[0].lower()
                    target_name = groups[-1].lower()  # Last group is usually the target
                    
                    # Find entity IDs
                    source_id = entity_map.get(source_name)
                    target_id = entity_map.get(target_name)
                    
                    if source_id and target_id and source_id != target_id:
                        relationship = Relationship(
                            source_id=source_id,
                            target_id=target_id,
                            type=rel_type,
                            properties={
                                "text": match.group(0),
                                "position": match.start()
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
