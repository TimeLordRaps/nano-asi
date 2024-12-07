"""Advanced configuration management system for NanoASI."""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
import yaml
import json
import importlib
import logging
from pathlib import Path
from .interfaces import ComponentConfig

logger = logging.getLogger(__name__)

class ConfigSource(BaseModel):
    """Configuration source definition."""
    type: str = Field(..., description="Source type (file, env, etc)")
    location: str = Field(..., description="Source location/path")
    format: str = Field(default='yaml', description="Config format")
    priority: int = Field(default=0, description="Loading priority")
    required: bool = Field(default=True, description="Whether source is required")

class ModularConfig:
    """Advanced configuration management with dynamic updates and validation."""
    
    def __init__(
        self,
        sources: Optional[List[ConfigSource]] = None,
        initial_config: Optional[Dict[str, Any]] = None
    ):
        self._config = initial_config or {}
        self._component_configs: Dict[str, ComponentConfig] = {}
        self._sources = sources or []
        self._load_history: List[Dict[str, Any]] = []
        self._watchers: Dict[str, List[callable]] = {}
        
        # Load configurations from sources
        if sources:
            self.load_from_sources()
    
    def load_from_sources(self):
        """Load and merge configurations from all sources."""
        for source in sorted(self._sources, key=lambda s: s.priority):
            try:
                config = self._load_source(source)
                self._merge_config(config)
                self._load_history.append({
                    'source': source.dict(),
                    'success': True
                })
            except Exception as e:
                logger.error(f"Failed to load config from {source.location}: {str(e)}")
                self._load_history.append({
                    'source': source.dict(),
                    'success': False,
                    'error': str(e)
                })
                if source.required:
                    raise
    
    def _load_source(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from a single source."""
        if source.type == 'file':
            return self._load_file(source.location, source.format)
        elif source.type == 'env':
            return self._load_env(source.location)
        else:
            raise ValueError(f"Unsupported config source type: {source.type}")
    
    def _load_file(self, path: str, format: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path) as f:
            if format == 'yaml':
                return yaml.safe_load(f)
            elif format == 'json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {format}")
    
    def _load_env(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        import os
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config[key[len(prefix):].lower()] = value
        return config
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing."""
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if isinstance(value, dict) and key in dict1:
                    dict1[key] = deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
            return dict1
        
        self._config = deep_merge(self._config, new_config)
        self._notify_watchers('config_updated', self._config)
    
    def add_component(
        self,
        name: str,
        component_config: ComponentConfig,
        validate: bool = True
    ):
        """Add or update a component configuration."""
        if validate:
            self._validate_component(component_config)
        self._component_configs[name] = component_config
        self._notify_watchers('component_added', name)
    
    def _validate_component(self, config: ComponentConfig):
        """Validate component configuration."""
        try:
            module_path, class_name = config.type.rsplit('.', 1)
            module = importlib.import_module(module_path)
            if not hasattr(module, class_name):
                raise ValueError(f"Class {class_name} not found in {module_path}")
        except (ImportError, ValueError) as e:
            raise ValueError(f"Invalid component type {config.type}: {str(e)}")
    
    def get_component_config(self, name: str) -> Optional[ComponentConfig]:
        """Retrieve a specific component's configuration."""
        return self._component_configs.get(name)
    
    def update(self, updates: Dict[str, Any], notify: bool = True):
        """Update configuration with new values."""
        self._merge_config(updates)
        if notify:
            self._notify_watchers('config_updated', updates)
    
    def watch(self, event: str, callback: callable):
        """Register a watcher for configuration changes."""
        if event not in self._watchers:
            self._watchers[event] = []
        self._watchers[event].append(callback)
    
    def _notify_watchers(self, event: str, data: Any):
        """Notify all watchers of an event."""
        for callback in self._watchers.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in config watcher callback: {str(e)}")
    
    def validate(self) -> bool:
        """Validate entire configuration."""
        try:
            # Validate component configurations
            for name, config in self._component_configs.items():
                self._validate_component(config)
            
            # Validate dependencies
            self._validate_dependencies()
            
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _validate_dependencies(self):
        """Validate component dependencies."""
        components = set(self._component_configs.keys())
        for name, config in self._component_configs.items():
            missing = set(config.dependencies) - components
            if missing:
                raise ValueError(
                    f"Component {name} has missing dependencies: {missing}"
                )
    
    def get_load_history(self) -> List[Dict[str, Any]]:
        """Get configuration loading history."""
        return self._load_history.copy()
