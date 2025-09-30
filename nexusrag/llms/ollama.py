from typing import List, Dict, Any
import os
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Language model implementation using Ollama for local LLMs."""
    
    def __init__(self, model_name: str = "llama2", host: str = "http://localhost:11434"):
        """Initialize the Ollama LLM.
        
        Args:
            model_name (str): Name of the Ollama model to use
            host (str): Ollama server host
        """
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "To use OllamaLLM, you need to install the ollama library. "
                "Please run: pip install ollama"
            )
        
        self.client = Client(host=host)
        self.model_name = model_name
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response using Ollama.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        # Build the full prompt with context
        if context:
            context_text = "\n".join([doc["content"] for doc in context])
            full_prompt = f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        # Generate response
        response = self.client.generate(
            model=self.model_name,
            prompt=full_prompt
        )
        return response['response']
