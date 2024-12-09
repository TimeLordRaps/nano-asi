"""Core architectural components for NanoASI."""

from typing import Protocol, runtime_checkable, Dict, Any, Optional

@runtime_checkable
class ComponentProtocol(Protocol):
    """Universal protocol for all NanoASI components."""
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        ...
    
    async def process(self, input_data: Any) -> Any:
        """Process input through the component."""
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """Retrieve current component state."""
        ...

class ComponentRegistry:
    """Centralized registry for managing and discovering components."""
    _components: Dict[str, ComponentProtocol] = {}
    
    @classmethod
    def register(cls, name: str, component: ComponentProtocol):
        """Register a new component."""
        cls._components[name] = component
    
    @classmethod
    def get(cls, name: str) -> Optional[ComponentProtocol]:
        """Retrieve a registered component."""
        return cls._components.get(name)
