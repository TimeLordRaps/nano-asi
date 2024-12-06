"""Example of using NanoASI to generate innovative climate change solutions."""

import asyncio
from nano_asi import ASI, Config

async def generate_climate_solution():
    """Generate an innovative solution for climate change."""
    config = Config(
        universe_exploration={"num_parallel_universes": 5},
        synthetic_data_config={
            "generation_strategies": [
                "domain_extrapolation", 
                "adversarial_generation", 
                "meta_learning_augmentation"
            ]
        }
    )
    
    asi = ASI(config=config)
    
    result = await asi.run(
        task="Generate a comprehensive and innovative solution for mitigating climate change",
        dataset="environmental_research_papers",
        stream=True
    )
    
    print("Climate Change Solution:")
    print(result.solution)
    print("\nMetrics:")
    print(result.metrics)
    print("\nConsciousness Flow:")
    print(result.consciousness_flow)

if __name__ == "__main__":
    asyncio.run(generate_climate_solution())
