"""Example of using NanoASI for advanced code generation."""

import asyncio
from nano_asi import ASI, Config
from nano_asi.modules import ConsciousnessTracker, LoRAGenerator

async def generate_code():
    """Generate code with consciousness-guided optimization."""
    config = Config(
        model_config={
            "base_model": "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
            "lora_rank": 256,
            "lora_alpha": 256
        },
        optimization_regime={
            "max_iterations": 100,
            "improvement_threshold": 0.1,
            "adaptive_exploration": True
        }
    )
    
    # Initialize with custom components
    asi = ASI(
        config=config,
        consciousness_tracker=ConsciousnessTracker(),
        lora_generator=LoRAGenerator()
    )
    
    # Example coding task
    task = """
    Create a Python implementation of a quantum-inspired neural network 
    with the following features:
    - Quantum state encoding layer
    - Quantum convolution operations
    - Entanglement-based attention mechanism
    - Measurement layer for classical output
    Include comprehensive documentation and example usage.
    """
    
    result = await asi.run(
        task=task,
        dataset="quantum_computing_papers",
        stream=True
    )
    
    print("Generated Code:")
    print(result.solution)
    print("\nConsciousness Flow Analysis:")
    print(result.consciousness_flow)
    print("\nPerformance Metrics:")
    print(result.metrics)

if __name__ == "__main__":
    asyncio.run(generate_code())
