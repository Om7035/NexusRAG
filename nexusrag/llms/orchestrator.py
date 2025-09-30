from typing import List, Dict, Any, Optional
from .base import BaseLLM
from .ollama import OllamaLLM
import subprocess
import os
import json


class LocalLLMOrchestrator:
    """Orchestrator for managing local LLMs with Ollama."""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """Initialize the local LLM orchestrator.
        
        Args:
            ollama_host (str): Ollama server host
        """
        self.ollama_host = ollama_host
        self.available_models = []
        self._check_ollama_status()
        self._load_available_models()
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _load_available_models(self) -> None:
        """Load available models from Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        self.available_models.append(model_name)
        except Exception:
            pass
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name (str): Name of the model to pull
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                if model_name not in self.available_models:
                    self.available_models.append(model_name)
                return True
            else:
                return False
        except Exception:
            return False
    
    def create_llm(self, model_name: str) -> BaseLLM:
        """Create an LLM instance for a specific model.
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            BaseLLM: LLM instance
        """
        # If model is not available, try to pull it
        if model_name not in self.available_models:
            if not self.pull_model(model_name):
                raise ValueError(f"Model '{model_name}' not available and could not be pulled")
        
        # Create Ollama LLM instance
        return OllamaLLM(model_name, host=self.ollama_host)
    
    def list_models(self) -> List[str]:
        """List available models.
        
        Returns:
            List[str]: List of available model names
        """
        return self.available_models.copy()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "name": model_name,
                    "available": True,
                    "details": result.stdout.strip()
                }
            else:
                return {
                    "name": model_name,
                    "available": False,
                    "error": result.stderr.strip()
                }
        except Exception as e:
            return {
                "name": model_name,
                "available": False,
                "error": str(e)
            }
