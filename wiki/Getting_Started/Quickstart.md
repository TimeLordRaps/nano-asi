# Quickstart Guide

Get up and running with NanoASI in minutes.

## Installation

```bash
pip install nano-asi
```

## Basic Usage

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

```python
from nano_asi.web import launch_ui

# Start the web interface
launch_ui()
```

## Configuration

```python
from nano_asi import ASI, Config

# Create custom configuration
config = Config(
    consciousness_tracking=True,
    parallel_universes=3,
    mcts_exploration_weight=1.5
)

# Initialize ASI with config
asi = ASI(config=config)
```

## Next Steps

- Explore [Basic Tutorials](../Tutorials/Basic_Usage.md)
- Learn about [Core Components](../Components/Overview.md)
- Check [Advanced Usage](../Tutorials/Advanced_Configuration.md)
