# WIP: Will be an open-webui aider unsloth recursive self-improvement framework.

# 🧠 NanoASI: Advanced Recursive Self-Improving AI Framework

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL3.0-yellow.svg)](https://github.com/TimeLordRaps/nano-asi/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

NanoASI is a user-friendly framework for building recursive self-improving AI systems. It combines advanced MCTS (Monte Carlo Tree Search), consciousness flow tracking, and parallel universe exploration with an emphasis on accessibility and ease of use.

## ✨ Features

- 🎯 **Simple to Start**: One line to install, one line to run
- 🧩 **Modular Design**: Use only what you need, extend what you want
- 🔄 **Self-Improving**: Continuous learning and optimization
- 🌐 **Universal Compatibility**: Works everywhere - local, cloud, or notebook
- 🎨 **Beautiful Interface**: Built-in web UI for visualization and control

## 🚀 Quick Start

### Installation

```bash
pip install nano-asi
```

### Basic Usage

```python
from nano_asi import ASI

# Initialize and run with defaults
asi = ASI()
result = await asi.run("Generate an innovative solution for climate change")
print(result.solution)
```

### Google Colab Usage

Run nano-asi directly in Google Colab:

```python
# Run setup script
!wget https://raw.githubusercontent.com/nanoasi/nano-asi/main/colab_setup.py
!python colab_setup.py

# Now you can use nano-asi
from nano_asi import ASI
import asyncio
import nest_asyncio
nest_asyncio.apply()

async def main():
    asi = ASI()
    result = await asi.run("Your task here")
    print(result.solution)

asyncio.run(main())
```

### Advanced Usage

```python
from nano_asi import ASI, Config
from nano_asi.modules import ConsciousnessTracker, LoRAGenerator

# Customize components
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

# Run with all features enabled
result = await asi.run(
    task="Generate an innovative solution for climate change",
    dataset="environmental_papers",
    stream=True  # Enable real-time output
)

# Access comprehensive results
print(result.solution)
print(result.consciousness_flow)
print(result.universe_explorations)
print(result.metrics)
```

### Web Interface

```python
from nano_asi.web import launch_ui

# Start the web interface
launch_ui()
```

## 🧩 Core Components

### For Beginners
- 🤖 **ASI Class**: Main interface for basic usage
- 🌐 **Web UI**: Visual interface for interaction
- 📊 **Basic Metrics**: Simple performance tracking

### For Intermediate Users
- 🎯 **MCTS Engine**: Guided exploration and optimization
- 🧬 **LoRA Diffusion**: Model adaptation and fine-tuning
- 📈 **Judgment System**: Multi-level evaluation framework

### For Advanced Users
- 🌌 **Universe Explorer**: Parallel optimization strategies
- 🧠 **Consciousness Tracker**: Deep pattern analysis
- 🔧 **Custom Components**: Build your own modules

## 📚 Documentation

- [Quick Start Guide](https://nano-asi.readthedocs.io/en/latest/quickstart.html)
- [Tutorial Notebooks](https://nano-asi.readthedocs.io/en/latest/tutorials/index.html)
- [API Reference](https://nano-asi.readthedocs.io/en/latest/api/index.html)
- [Advanced Topics](https://nano-asi.readthedocs.io/en/latest/advanced/index.html)

## 🛠️ Development

```bash
# Install development dependencies
pip install nano-asi[dev]

# Run tests
pytest

# Build documentation
pip install nano-asi[docs]
cd docs && make html
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

GPL-3.0 License - see [LICENSE](https://github.com/TimeLordRaps/nano-asi/blob/main/LICENSE) for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nanoasi/nano-asi&type=Date)](https://star-history.com/#nanoasi/nano-asi&Date)

## 📫 Contact

🐦 Twitter/X: [@TimeLordRaps](https://x.com/TimeLordRaps)
