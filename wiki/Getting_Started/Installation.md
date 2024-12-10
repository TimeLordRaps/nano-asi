# Installation Guide for NanoASI

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Optional: CUDA-compatible GPU for enhanced performance

## Installation Methods

### 1. Standard Installation

Install the latest stable version using pip:

```bash
pip install nano-asi
```

### 2. Development Installation

For the latest development version:

```bash
git clone https://github.com/TimeLordRaps/nano-asi.git
cd nano-asi
pip install -e .
```

### 3. Conda Environment (Recommended)

Create a dedicated conda environment:

```bash
conda create -n nanoasi python=3.10
conda activate nanoasi
pip install nano-asi
```

## Optional Dependencies

### GPU Support
For GPU acceleration:

```bash
pip install nano-asi[gpu]
```

### Development Tools
For development and testing:

```bash
pip install nano-asi[dev]
```

## Verify Installation

```python
import nano_asi

# Check version
print(nano_asi.__version__)

# Quick test
from nano_asi import ASI
import asyncio

async def test():
    asi = ASI()
    result = await asi.run("Hello, NanoASI!")
    print(result)

asyncio.run(test())
```

## Troubleshooting

- Ensure you have the latest pip: `pip install --upgrade pip`
- Check Python version compatibility
- Verify CUDA installation for GPU support

## Next Steps

- [Quickstart Guide](Quickstart.md)
- [Basic Usage Tutorial](../Tutorials/Basic_Usage.md)
- [Configuration Guide](../Advanced/Configuration.md)
