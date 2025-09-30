import pytest
from unittest.mock import patch, MagicMock


def test_ollama_llm_import():
    """Test that the Ollama LLM can be imported."""
    from nexusrag.llms.ollama import OllamaLLM
    assert OllamaLLM


def test_ollama_llm_initialization():
    """Test that the Ollama LLM can be initialized."""
    try:
        from nexusrag.llms.ollama import OllamaLLM
        llm = OllamaLLM()
        assert llm
    except ImportError:
        # If ollama is not installed, skip this test
        pytest.skip("ollama not installed")


def test_universal_llm_with_ollama():
    """Test that the universal LLM can use Ollama."""
    try:
        from nexusrag.llms.universal import UniversalLLM
        llm = UniversalLLM(provider="ollama")
        assert llm.provider == "ollama"
    except ImportError:
        # If ollama is not installed, skip this test
        pytest.skip("ollama not installed")
