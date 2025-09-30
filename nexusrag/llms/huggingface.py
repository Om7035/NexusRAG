from typing import List, Dict, Any
from .base import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """Language model implementation using Hugging Face Transformers."""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize the LLM with a specific model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "To use HuggingFaceLLM, you need to install the transformers library. "
                "Please run: pip install transformers"
            )
        
        self.model_name = model_name
        self.pipeline = pipeline("text2text-generation", model=model_name)
    
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response based on a prompt and optional context.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        # If context is provided, include it in the prompt
        if context:
            context_texts = [doc["content"] for doc in context]
            context_str = "\n".join(context_texts)
            full_prompt = f"Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        # Generate response
        result = self.pipeline(full_prompt, max_length=200, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]
