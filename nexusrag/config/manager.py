import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Configuration manager for NexusRAG."""
    
    def __init__(self, config_path: str = None):
        """Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Determine config file path
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config in several locations
            possible_paths = [
                Path.cwd() / "nexusrag.yaml",
                Path.cwd() / "nexusrag.yml",
                Path.home() / ".nexusrag" / "config.yaml",
                Path(__file__).parent / "default.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.config_path = path
                    break
            else:
                # Use default config
                self.config_path = Path(__file__).parent / "default.yaml"
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            print("Using default configuration...")
            # Load default config
            default_path = Path(__file__).parent / "default.yaml"
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key (str): Configuration key (e.g., "pipeline.chunk_size")
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """Save configuration to file.
        
        Args:
            path (str): Path to save configuration (uses config_path if None)
        """
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        return self.config.get("pipeline", {})
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get component configuration.
        
        Args:
            component (str): Component name
            
        Returns:
            Dict[str, Any]: Component configuration
        """
        return self.config.get("components", {}).get(component, {})
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced features configuration.
        
        Returns:
            Dict[str, Any]: Advanced features configuration
        """
        return self.config.get("advanced", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Dict[str, Any]: Logging configuration
        """
        return self.config.get("logging", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration.
        
        Returns:
            Dict[str, Any]: Performance configuration
        """
        return self.config.get("performance", {})
    
    def __str__(self) -> str:
        """String representation of the configuration.
        
        Returns:
            str: String representation
        """
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration.
        
        Returns:
            str: Detailed representation
        """
        return f"ConfigManager(config_path={self.config_path}, config={self.config})"
