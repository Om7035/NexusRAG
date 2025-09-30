import pytest
import os
from unittest.mock import patch, MagicMock


def test_gemini_embedder_import():
    """Test that the Gemini embedder can be imported."""
    from nexusrag.embedders.gemini import GeminiEmbedder
    assert GeminiEmbedder


def test_gemini_llm_import():
    """Test that the Gemini LLM can be imported."""
    from nexusrag.llms.gemini import GeminiLLM
    assert GeminiLLM


def test_gemini_embedder_initialization():
    """Test that the Gemini embedder can be initialized."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        try:
            from nexusrag.embedders.gemini import GeminiEmbedder
            embedder = GeminiEmbedder()
            assert embedder
        except ImportError:
            # If google-generativeai is not installed, skip this test
            pytest.skip("google-generativeai not installed")


def test_gemini_llm_initialization():
    """Test that the Gemini LLM can be initialized."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        try:
            from nexusrag.llms.gemini import GeminiLLM
            llm = GeminiLLM()
            assert llm
        except ImportError:
            # If google-generativeai is not installed, skip this test
            pytest.skip("google-generativeai not installed")


def test_universal_embedder_with_gemini():
    """Test that the universal embedder can use Gemini."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        try:
            from nexusrag.embedders.universal import UniversalEmbedder
            embedder = UniversalEmbedder(provider="gemini")
            assert embedder.provider == "gemini"
        except ImportError:
            # If google-generativeai is not installed, skip this test
            pytest.skip("google-generativeai not installed")


def test_universal_llm_with_gemini():
    """Test that the universal LLM can use Gemini."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        try:
            from nexusrag.llms.universal import UniversalLLM
            llm = UniversalLLM(provider="gemini")
            assert llm.provider == "gemini"
        except ImportError:
            # If google-generativeai is not installed, skip this test
            pytest.skip("google-generativeai not installed")
