"""Core configuration module for nano-asi."""

from typing import Dict, Any
import os

class Config:
    """
    Central configuration class for nano-asi.
    
    Provides a flexible configuration management system with 
    default and customizable settings.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration with optional custom config path.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        # Default configuration
        self.model_config = {
            'max_seq_length': 128000,
            'model_name': 'default_model',
            'device': 'auto',
            'precision': 'float16'
        }
        
        # If a config path is provided, load custom configurations
        if config_path and os.path.exists(config_path):
            self._load_config_from_file(config_path)
    
    def _load_config_from_file(self, config_path: str):
        """
        Load configuration from a file.
        
        Currently supports basic dictionary-based configuration.
        Can be extended to support JSON, YAML, etc.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            # Placeholder for more advanced config loading
            # For now, just a simple dict update
            with open(config_path, 'r') as f:
                custom_config = eval(f.read())
                self.model_config.update(custom_config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
        
        Returns:
            Configuration value or default
        """
        return self.model_config.get(key, default)
    
    def update(self, config: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            config: Dictionary of configuration updates
        """
        self.model_config.update(config)
