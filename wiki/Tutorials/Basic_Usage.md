# Basic Usage Tutorial

## Installation

Install NanoASI using pip:

```bash
pip install nano-asi
```

## Basic ASI Initialization

The simplest way to use NanoASI is to create an ASI instance with default settings:

```python
from nano_asi import ASI

# Initialize with defaults
asi = ASI()
```

## Running Your First Task

Here's a simple example of running a task:

```python
import asyncio
from nano_asi import ASI

async def main():
    asi = ASI()
    
    # Run a simple task
    result = await asi.run("Generate an innovative solution for climate change")
    
    # Print the solution
    print(result.solution)
    
    # Access additional metrics
    print("\nMetrics:")
    print(result.metrics)

# Run the async function
asyncio.run(main())
```

## Understanding Results

The `ASIResult` object contains:
- `solution`: The generated solution text
- `consciousness_flow`: Tracked consciousness states
- `universe_explorations`: Results from parallel universes
- `metrics`: Performance and temporal metrics

Example of accessing detailed results:

```python
# Access consciousness flow
print("\nConsciousness Flow:")
for state in result.consciousness_flow:
    print(f"- State at {state.timestamp}")
    print(f"  Patterns: {len(state.activation_patterns)}")
    print(f"  Thoughts: {len(state.thought_chains)}")

# Access universe explorations
print("\nUniverse Explorations:")
for universe_id, data in result.universe_explorations.items():
    print(f"Universe {universe_id}: {data['score']}")
```

## Simple Configuration

Basic configuration can be done using the Config class:

```python
from nano_asi import ASI, Config

# Create custom configuration
config = Config(
    consciousness_tracking=True,  # Enable consciousness tracking
    parallel_universes=3,         # Number of parallel universes to explore
    mcts_exploration_weight=1.5   # Exploration weight for MCTS
)

# Initialize ASI with config
asi = ASI(config=config)
```

## Next Steps

- Explore [Advanced Configuration](Advanced_Configuration.md)
- Learn about [Custom Components](Custom_Components.md)
- Check the [API Reference](../API_Reference/Overview.md)
