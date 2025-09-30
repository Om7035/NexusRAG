import pytest


def test_package_import():
    """Test that the nexusrag package can be imported."""
    import nexusrag
    assert nexusrag.__version__ == "0.1.0"


def test_module_imports():
    """Test that all main modules can be imported."""
    from nexusrag import pipeline
    from nexusrag.parsers import base
    from nexusrag.embedders import base
    from nexusrag.vectorstores import base
    from nexusrag.llms import base
    
    # Test that specific classes can be imported
    from nexusrag.pipeline import RAGPipeline
    from nexusrag.parsers.base import BaseParser, Document
    from nexusrag.embedders.base import BaseEmbedder
    from nexusrag.vectorstores.base import BaseVectorStore
    from nexusrag.llms.base import BaseLLM


def test_component_imports():
    """Test that all component implementations can be imported."""
    from nexusrag.parsers.pdf import PDFParser
    from nexusrag.embedders.sentence_transformers import SentenceTransformerEmbedder
    from nexusrag.vectorstores.chroma import ChromaVectorStore
    from nexusrag.llms.huggingface import HuggingFaceLLM
