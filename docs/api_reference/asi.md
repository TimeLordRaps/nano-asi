# ASI Class Reference

## Overview

The `ASI` class is the main interface for the NanoASI framework, implementing recursive self-improvement with consciousness-guided optimization.

## Initialization

```python
from nano_asi import ASI, Config

asi = ASI(
    config: Optional[Config] = None,
    consciousness_tracker: Optional[ConsciousnessTracker] = None,
    lora_generator: Optional[LoRAGenerator] = None,
    mcts_engine: Optional[MCTSEngine] = None,
    judgment_system: Optional[JudgmentSystem] = None,
    universe_explorer: Optional[UniverseExplorer] = None
)
```

### Parameters

- `config`: Configuration object controlling system behavior
- `consciousness_tracker`: Component for tracking consciousness flow
- `lora_generator`: Component for LoRA adapter generation
- `mcts_engine`: Monte Carlo Tree Search engine
- `judgment_system`: Hierarchical judgment system
- `universe_explorer`: Parallel universe exploration component

## Core Methods

### run()

```python
async def run(
    task: str,
    dataset: Optional[Union[str, Dataset]] = None,
    stream: bool = False,
    max_iterations: Optional[int] = None
) -> ASIResult
```

Executes a recursive self-improvement cycle with temporal investment tracking.

#### Parameters

- `task`: Primary task or prompt
- `dataset`: Optional dataset for context/training
- `stream`: Enable real-time output
- `max_iterations`: Override default iteration limit

#### Returns

`ASIResult` object containing:
- `solution`: Generated solution
- `consciousness_flow`: Consciousness tracking data
- `universe_explorations`: Parallel universe results
- `metrics`: Performance and temporal metrics

## Configuration Options

The ASI class behavior can be customized through the Config object:

```python
config = Config(
    universe_exploration={
        "num_parallel_universes": 5,
        "exploration_strategy": "adaptive_mcts"
    },
    token_investment={
        "base_learning_rate": 0.01,
        "meta_learning_enabled": True
    }
)
```

## Advanced Usage

### Consciousness Flow Integration

```python
# Enable consciousness tracking
consciousness_tracker = ConsciousnessTracker()
asi = ASI(consciousness_tracker=consciousness_tracker)

# Run with consciousness flow
result = await asi.run(
    task="Generate innovative solution",
    stream=True  # See consciousness flow in real-time
)
print(result.consciousness_flow)
```

### Parallel Universe Exploration

```python
# Configure universe exploration
config = Config(
    universe_exploration={
        "num_parallel_universes": 10,
        "cross_universe_coherence": 0.7
    }
)

# Run with parallel exploration
result = await asi.run(task="Complex optimization task")
print(result.universe_explorations)
```

### Token Investment Tracking

```python
# Access temporal metrics
print(result.metrics['tokens_invested'])
print(result.metrics['temporal_roi'])
```

## Result Structure

The `ASIResult` object provides comprehensive information:

```python
class ASIResult(BaseModel):
    solution: str  # Generated solution
    consciousness_flow: Optional[Dict[str, Any]]  # Consciousness tracking
    universe_explorations: Optional[Dict[str, Any]]  # Universe results
    metrics: Optional[Dict[str, Any]]  # Performance metrics
```

### Metrics Include:

- `tokens_invested`: Total tokens used
- `temporal_roi`: Return on temporal investment
- `improvement_trajectory`: Solution improvement path
- `consciousness_coherence`: Flow coherence metrics
- `universe_diversity`: Cross-universe pattern diversity

## Best Practices

1. **Token Investment**
   - Monitor token usage with `metrics['tokens_invested']`
   - Use streaming for real-time optimization tracking

2. **Consciousness Flow**
   - Enable consciousness tracking for complex tasks
   - Analyze flow patterns in `consciousness_flow`

3. **Universe Exploration**
   - Adjust `num_parallel_universes` based on task complexity
   - Monitor cross-universe coherence

4. **Performance Optimization**
   - Use appropriate `max_iterations` for your task
   - Monitor improvement trajectory

## Examples

### Basic Usage

```python
from nano_asi import ASI

asi = ASI()
result = await asi.run("Generate an innovative solution")
print(result.solution)
```

### Advanced Configuration

```python
from nano_asi import ASI, Config
from nano_asi.modules import ConsciousnessTracker, LoRAGenerator

config = Config(
    consciousness_tracking=True,
    parallel_universes=3,
    mcts_exploration_weight=1.5
)

asi = ASI(
    config=config,
    consciousness_tracker=ConsciousnessTracker(),
    lora_generator=LoRAGenerator()
)

result = await asi.run(
    task="Complex optimization task",
    dataset="specialized_dataset",
    stream=True
)
```

### Analyzing Results

```python
# Access solution
print(result.solution)

# Analyze consciousness flow
print(result.consciousness_flow)

# Check universe explorations
print(result.universe_explorations)

# Review metrics
print(result.metrics)
```
