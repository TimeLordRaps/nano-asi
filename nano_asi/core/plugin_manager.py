"""Dynamic plugin discovery and management for NanoASI."""

import importlib
import pkgutil
import inspect
from typing import Dict, Type, Any, Optional
from pathlib import Path

from .interfaces import ComponentInterface, ComponentConfig

class PluginManager:
    """Dynamic plugin discovery and management.
    
    Handles:
    - Plugin discovery and loading
    - Plugin lifecycle management
    - Plugin configuration validation
    """
    
    @staticmethod
    def discover_plugins(package_name: str) -> Dict[str, Type[ComponentInterface]]:
        """Discover and load plugins from a given package.
        
        Args:
            package_name: Name of package to search for plugins
            
        Returns:
            Dictionary mapping plugin names to plugin classes
        """
        plugins = {}
        package = importlib.import_module(package_name)
        
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    hasattr(obj, 'is_plugin') and
                    isinstance(obj.is_plugin, bool) and 
                    obj.is_plugin):
                    plugins[name] = obj
        
        return plugins

    @staticmethod
    async def load_plugin(
        plugin_class: Type[ComponentInterface],
        config: ComponentConfig
    ) -> Optional[ComponentInterface]:
        """Load and initialize a specific plugin.
        
        Args:
            plugin_class: Plugin class to instantiate
            config: Configuration for the plugin
            
        Returns:
            Initialized plugin instance or None if initialization fails
        """
        try:
            plugin = plugin_class()
            await plugin.initialize(config.parameters)
            return plugin
        except Exception as e:
            print(f"Failed to load plugin {plugin_class.__name__}: {str(e)}")
            return None

    @staticmethod
    def validate_plugin(plugin_class: Type[Any]) -> bool:
        """Validate that a plugin implements required interfaces.
        
        Args:
            plugin_class: Class to validate
            
        Returns:
            True if plugin is valid, False otherwise
        """
        required_methods = {
            'initialize',
            'process',
            'get_state'
        }
        
        return all(
            hasattr(plugin_class, method) 
            for method in required_methods
        )
