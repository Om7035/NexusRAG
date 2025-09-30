from typing import List, Dict, Any
from .base import BaseLLM


class UniversalLLM(BaseLLM):
    """Universal LLM that can use different language model providers."""
    
    def __init__(self, provider: str = "huggingface", model_name: str = None):
        """Initialize the universal LLM.
        
        Args:
            provider (str): LLM provider ("huggingface", "openai", "anthropic", "gemini", "ollama")
            model_name (str): Specific model name to use
        """
        self.provider = provider.lower()
        
        if self.provider == "huggingface":
            from .huggingface import HuggingFaceLLM
            model_name = model_name or "google/flan-t5-base"
            self.llm = HuggingFaceLLM(model_name)
        elif self.provider == "openai":
            from .openai import OpenAILLM
            model_name = model_name or "gpt-3.5-turbo"
            self.llm = OpenAILLM(model_name)
        elif self.provider == "anthropic":
            from .anthropic import AnthropicLLM
            model_name = model_name or "claude-3-haiku-20240307"
            self.llm = AnthropicLLM(model_name)
        elif self.provider == "gemini":
            from .gemini import GeminiLLM
            model_name = model_name or "gemini-pro"
            self.llm = GeminiLLM(model_name)
        elif self.provider == "ollama":
            from .ollama import OllamaLLM
            model_name = model_name or "llama2"
            self.llm = OllamaLLM(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response using the selected LLM.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        return self.llm.generate(prompt, context)
