from typing import List, Dict, Any
import os
from .base import BaseLLM


class GeminiLLM(BaseLLM):
    """Language model implementation using Google Gemini's API."""
    
    def __init__(self, model_name: str = "gemini-pro"):
        """Initialize the Gemini LLM.
        
        Args:
            model_name (str): Name of the Gemini model to use
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "To use GeminiLLM, you need to install the google-generativeai library. "
                "Please run: pip install google-generativeai"
            )
        
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
            )
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response using Gemini's API.
        
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
        response = self.model.generate_content(full_prompt)
        return response.text
