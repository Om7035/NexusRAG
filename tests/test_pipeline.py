import pytest
from unittest.mock import Mock, patch
from nexusrag.pipeline import RAGPipeline
from nexusrag.parsers.base import Document


def test_pipeline_initialization():
    """Test that the pipeline can be initialized with components."""
    # Create mock components
    mock_parser = Mock()
    mock_embedder = Mock()
    mock_vector_store = Mock()
    mock_llm = Mock()
    
    # Initialize pipeline
    pipeline = RAGPipeline(mock_parser, mock_embedder, mock_vector_store, mock_llm)
    
    # Assert that components are set correctly
    assert pipeline.parser == mock_parser
    assert pipeline.embedder == mock_embedder
    assert pipeline.vector_store == mock_vector_store
    assert pipeline.llm == mock_llm


def test_process_document():
    """Test that process_document calls the parser and vector store correctly."""
    # Create mock components
    mock_parser = Mock()
    mock_embedder = Mock()
    mock_vector_store = Mock()
    mock_llm = Mock()
    
    # Create pipeline
    pipeline = RAGPipeline(mock_parser, mock_embedder, mock_vector_store, mock_llm)
    
    # Mock parser response
    mock_documents = [Document("Test content", {"source": "test.pdf"})]
    mock_parser.parse.return_value = mock_documents
    
    # Call process_document
    test_file_path = "test.pdf"
    pipeline.process_document(test_file_path)
    
    # Assert parser was called correctly
    mock_parser.parse.assert_called_once_with(test_file_path)
    
    # Assert vector store was called correctly
    mock_vector_store.add.assert_called_once_with(mock_documents)


def test_ask():
    """Test that ask calls the vector store and LLM correctly."""
    # Create mock components
    mock_parser = Mock()
    mock_embedder = Mock()
    mock_vector_store = Mock()
    mock_llm = Mock()
    
    # Create pipeline
    pipeline = RAGPipeline(mock_parser, mock_embedder, mock_vector_store, mock_llm)
    
    # Mock vector store response
    mock_context = [{"content": "Test context", "score": 0.9}]
    mock_vector_store.query.return_value = mock_context
    
    # Mock LLM response
    mock_answer = "This is a test answer."
    mock_llm.generate.return_value = mock_answer
    
    # Call ask
    test_question = "What is this about?"
    answer = pipeline.ask(test_question)
    
    # Assert vector store was called correctly
    mock_vector_store.query.assert_called_once_with(test_question)
    
    # Assert LLM was called correctly
    mock_llm.generate.assert_called_once_with(test_question, mock_context)
    
    # Assert correct answer was returned
    assert answer == mock_answer
