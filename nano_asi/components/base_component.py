"""Base component implementation for NanoASI."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.architecture import ComponentProtocol

class BaseComponent(ComponentProtocol, ABC):
    """Base class for all NanoASI components with enhanced traceability."""
    
    def __init__(self, name: str):
        self.name = name
        self._state_history: list = []
        self._performance_metrics: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input through the component."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Retrieve current component state with performance tracking."""
        return {
            'name': self.name,
            'current_state': self._state_history[-1] if self._state_history else None,
            'performance_metrics': self._performance_metrics
        }
    
    def _record_state(self, state: Any):
        """Record state for traceability."""
        self._state_history.append(state)
