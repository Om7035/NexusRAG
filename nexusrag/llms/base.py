from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLM(ABC):
    """Abstract base class for large language models."""
    
    @abstractmethod
    def generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response based on a prompt and optional context.
        
        Args:
            prompt (str): The prompt to generate a response for
            context (List[Dict[str, Any]]): Optional context documents
            
        Returns:
            str: Generated response
        """
        pass
