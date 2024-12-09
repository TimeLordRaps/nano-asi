"""Core interfaces and protocols for NanoASI components."""

from typing import Protocol, Dict, Any, Optional, List, Union, TypeVar, Generic
from pydantic import BaseModel, Field
import torch

T = TypeVar('T')

class ComponentProtocol(Protocol):
    """Base protocol for all NanoASI components."""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with given configuration."""
        ...

    async def process(self, input_data: Any) -> Any:
        """Process input data through the component."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Retrieve current component state."""
        ...
    
    async def validate(self) -> bool:
        """Validate component state and configuration."""
        ...

class ModelAdapterProtocol(ComponentProtocol):
    """Protocol for model adaptation and generation."""
    
    async def generate(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate output based on prompt and optional context."""
        ...

    async def fine_tune(
        self, 
        dataset: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fine-tune the model on a given dataset."""
        ...
    
    async def get_embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """Get embeddings for input text."""
        ...

class JudgmentProtocol(ComponentProtocol):
    """Protocol for hierarchical judgment systems."""
    
    async def evaluate(
        self, 
        input_data: Any, 
        criteria: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Evaluate input data against specified criteria."""
        ...
    
    async def compare(
        self,
        candidate1: Any,
        candidate2: Any,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compare two candidates with detailed metrics."""
        ...
    
    async def aggregate_judgments(
        self,
        judgments: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate multiple judgments into a final score."""
        ...

class ComponentConfig(BaseModel):
    """Standardized configuration for components."""
    
    name: str = Field(..., description="Unique name of the component")
    type: str = Field(..., description="Full import path of component class")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Component parameters")
    enabled: bool = Field(default=True, description="Whether component is enabled")
    logging_level: str = Field(default='INFO', description="Component logging level")
    dependencies: List[str] = Field(default_factory=list, description="Required component dependencies")
    version: str = Field(default='1.0.0', description="Component version")
