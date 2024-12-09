# Quickstart Guide

## Installation

Install NanoASI using pip:

```bash
pip install nano-asi
```

## Basic Usage

Here's a simple example to get started:

```python
from nano_asi import ASI
import asyncio

async def main():
    # Initialize ASI
    asi = ASI()
    
    # Run a task
    result = await asi.run("Generate an innovative solution for climate change")
    print(result.solution)

# Run the async function
asyncio.run(main())
```

## Web Interface

Launch the web UI with:

```python
from nano_asi.web import launch_ui

launch_ui()
```

## Advanced Configuration

```python
from nano_asi import ASI, Config
from nano_asi.modules import ConsciousnessTracker, LoRAGenerator

# Customize configuration
config = Config(
    consciousness_tracking=True,
    parallel_universes=3,
    mcts_exploration_weight=1.5
)

# Initialize with custom components
asi = ASI(
    config=config,
    consciousness_tracker=ConsciousnessTracker(),
    lora_generator=LoRAGenerator()
)

# Run with advanced features
result = await asi.run(
    task="Solve a complex problem",
    dataset="specialized_dataset",
    stream=True
)
```

## Next Steps
- [Explore Tutorials](tutorials/)
- [View API Reference](api_reference/)
- [Learn Advanced Topics](advanced_topics/)
