# Quickstart Guide

Get up and running with NanoASI in minutes.

## Installation

```bash
pip install nano-asi
```

## Basic Usage

```python
from nano_asi import ASI

# Initialize and run with defaults
asi = ASI()
result = await asi.run("Generate an innovative solution for climate change")
print(result.solution)
```

## Web Interface

```python
from nano_asi.web import launch_ui

# Start the web interface
launch_ui()
```

## Next Steps

- Explore [Basic Tutorials](Tutorials/Basic_Usage)
- Learn about [Core Components](Components/ASI_Class)
- Check [Advanced Usage](Tutorials/Advanced_Usage)
