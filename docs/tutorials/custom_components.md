# Creating Custom Components

## Implementing ComponentProtocol

Custom components must implement the `ComponentProtocol`:

```python
from typing import Dict, Any
from nano_asi.core.interfaces import ComponentProtocol

class MyCustomComponent(ComponentProtocol):
    """Custom component implementation."""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize component with configuration."""
        self.config = config
        self.state = {}
    
    async def process(self, input_data: Any) -> Any:
        """Process input through the component."""
        # Add your processing logic here
        return processed_result
    
    def get_state(self) -> Dict[str, Any]:
        """Retrieve current component state."""
        return self.state
```

## Integrating with ASI

Register and use your custom component:

```python
from nano_asi import ASI, Config

# Initialize your component
my_component = MyCustomComponent()

# Create ASI instance with custom component
asi = ASI(
    config=Config(),
    custom_component=my_component
)
```

## Best Practices

1. **Clear Documentation**
   - Document your component's purpose
   - Explain configuration options
   - Provide usage examples

2. **Error Handling**
   ```python
   async def process(self, input_data: Any) -> Any:
       try:
           # Processing logic
           return result
       except Exception as e:
           logger.error(f"Error in component: {str(e)}")
           raise
   ```

3. **State Management**
   ```python
   class MyComponent(ComponentProtocol):
       def __init__(self):
           self._state_history = []
           self._performance_metrics = {}
       
       def get_state(self) -> Dict[str, Any]:
           return {
               'current_state': self._state_history[-1] if self._state_history else None,
               'metrics': self._performance_metrics
           }
   ```

4. **Configuration Validation**
   ```python
   async def initialize(self, config: Dict[str, Any]) -> None:
       if 'required_param' not in config:
           raise ValueError("Missing required parameter")
       self.config = config
   ```

## Example Custom Component

Here's a complete example of a custom consciousness enhancement component:

```python
from typing import Dict, Any, List
from nano_asi.core.interfaces import ComponentProtocol
import torch
import numpy as np
from collections import defaultdict

class ConsciousnessEnhancer(ComponentProtocol):
    """Custom component for enhancing consciousness flow."""
    
    def __init__(self):
        self.pattern_history = []
        self.enhancement_metrics = defaultdict(list)
        self.quantum_states = []
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config
        self.enhancement_factor = config.get('enhancement_factor', 1.0)
        self.quantum_depth = config.get('quantum_depth', 3)
    
    async def process(self, consciousness_state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance consciousness state."""
        # Extract patterns
        patterns = self._extract_patterns(consciousness_state)
        
        # Apply quantum enhancement
        enhanced_patterns = self._quantum_enhance(patterns)
        
        # Update state
        enhanced_state = consciousness_state.copy()
        enhanced_state['patterns'] = enhanced_patterns
        enhanced_state['enhancement_metrics'] = self._compute_metrics(patterns, enhanced_patterns)
        
        # Record history
        self.pattern_history.append(enhanced_patterns)
        
        return enhanced_state
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'pattern_history_length': len(self.pattern_history),
            'enhancement_metrics': dict(self.enhancement_metrics),
            'quantum_states': self.quantum_states
        }
    
    def _extract_patterns(self, state: Dict[str, Any]) -> List[Any]:
        """Extract consciousness patterns."""
        if 'patterns' in state:
            return state['patterns']
        return []
    
    def _quantum_enhance(self, patterns: List[Any]) -> List[Any]:
        """Apply quantum enhancement to patterns."""
        enhanced = []
        for pattern in patterns:
            # Convert to tensor
            if not isinstance(pattern, torch.Tensor):
                pattern = torch.tensor(pattern)
            
            # Apply quantum transformation
            enhanced_pattern = self._apply_quantum_transform(pattern)
            enhanced.append(enhanced_pattern)
        
        return enhanced
    
    def _apply_quantum_transform(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply quantum transformation to pattern."""
        # Simulate quantum enhancement
        transformed = pattern
        for _ in range(self.quantum_depth):
            transformed = torch.tanh(transformed * self.enhancement_factor)
        return transformed
    
    def _compute_metrics(
        self,
        original: List[Any],
        enhanced: List[Any]
    ) -> Dict[str, float]:
        """Compute enhancement metrics."""
        return {
            'enhancement_factor': self.enhancement_factor,
            'pattern_count': len(enhanced),
            'quantum_depth': self.quantum_depth,
            'average_enhancement': np.mean([
                torch.norm(e).item() / torch.norm(torch.tensor(o)).item()
                for e, o in zip(enhanced, original)
            ])
        }
```

## Using Custom Components with Dependency Injection

Leverage the dependency injection system:

```python
from nano_asi.core.dependency_injection import inject

class EnhancedASI:
    @inject('consciousness_enhancer')
    async def process_with_enhancement(
        self,
        task: str,
        consciousness_enhancer: ConsciousnessEnhancer
    ) -> Dict[str, Any]:
        # Process with injected component
        result = await self.run(task)
        enhanced = await consciousness_enhancer.process(result.consciousness_flow)
        return enhanced
```

## Next Steps

- Review the [API Reference](../api_reference/) for detailed interface documentation
- Explore [Example Components](../../examples/) for more inspiration
- Join the community to share your components
