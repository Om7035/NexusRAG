from typing import List, Dict, Any
import os
from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Language model implementation using Anthropic's API."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """Initialize the Anthropic LLM.
        
        Args:
            model_name (str): Name of the Anthropic model to use
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "To use AnthropicLLM, you need to install the anthropic library. "
                "Please run: pip install anthropic"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable."
            )
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response using Anthropic's API.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        # Build the prompt
        if context:
            context_text = "\n".join([doc["content"] for doc in context])
            full_prompt = f"Human: Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"
        
        # Generate response
        response = self.client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            temperature=0.7,
            max_tokens_to_sample=500
        )
        
        return response.completion
