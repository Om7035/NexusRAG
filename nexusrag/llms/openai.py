from typing import List, Dict, Any
import os
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """Language model implementation using OpenAI's API."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI LLM.
        
        Args:
            model_name (str): Name of the OpenAI model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "To use OpenAILLM, you need to install the openai library. "
                "Please run: pip install openai"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response using OpenAI's API.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        # Build the messages
        messages = []
        
        # Add context if provided
        if context:
            context_text = "\n".join([doc["content"] for doc in context])
            messages.append({
                "role": "system",
                "content": f"Use the following context to answer the question:\n\n{context_text}"
            })
        
        # Add the user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
