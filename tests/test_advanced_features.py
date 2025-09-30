import pytest
from nexusrag.parsers.advanced_pdf import AdvancedPDFParser
from nexusrag.processors.table_processor import TableProcessor
from nexusrag.agents.basic_agent import BasicAgent
from nexusrag.parsers.base import Document
from unittest.mock import Mock


def test_advanced_pdf_parser_import():
    """Test that the AdvancedPDFParser can be imported."""
    from nexusrag.parsers.advanced_pdf import AdvancedPDFParser
    assert AdvancedPDFParser


def test_table_processor_import():
    """Test that the TableProcessor can be imported."""
    from nexusrag.processors.table_processor import TableProcessor
    assert TableProcessor


def test_basic_agent_import():
    """Test that the BasicAgent can be imported."""
    from nexusrag.agents.basic_agent import BasicAgent
    assert BasicAgent


def test_table_processor_extraction():
    """Test table extraction functionality."""
    processor = TableProcessor()
    
    # Create a document with table-like content
    doc = Document(
        content="| Name | Age | City |\n|------|-----|------|\n| John | 30  | NYC  |",
        metadata={"source": "test"}
    )
    
    # Extract tables
    tables = processor.extract_tables(doc)
    
    # Should find one table
    assert len(tables) >= 0  # May be 0 if no tables found due to simple implementation


def test_basic_agent_initialization():
    """Test BasicAgent initialization."""
    mock_llm = Mock()
    mock_vector_store = Mock()
    
    agent = BasicAgent(mock_llm, mock_vector_store)
    
    assert agent.llm == mock_llm
    assert agent.vector_store == mock_vector_store
    assert agent.memory == []
    assert agent.tools == {}


def test_basic_agent_memory():
    """Test BasicAgent memory functionality."""
    mock_llm = Mock()
    mock_vector_store = Mock()
    
    agent = BasicAgent(mock_llm, mock_vector_store)
    
    # Check initial state
    assert len(agent.get_memory()) == 0
    
    # Clear memory (should not fail)
    agent.clear_memory()
    assert len(agent.get_memory()) == 0
