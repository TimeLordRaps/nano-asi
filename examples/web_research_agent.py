"""Example of using NanoASI for advanced web research."""

import asyncio
from nano_asi import ASI, Config
from nano_asi.modules import MCTSEngine, SyntheticDataGenerator

async def web_researcher():
    """Run NanoASI as an advanced web research agent."""
    config = Config(
        hypertraining_config={
            "max_boost_cycles": 10,
            "learning_acceleration_factor": 1.5,
            "meta_learning_enabled": True
        },
        synthetic_data_config={
            "generation_strategies": [
                "domain_extrapolation",
                "adversarial_generation",
                "meta_learning_augmentation"
            ],
            "diversity_threshold": 0.8
        }
    )
    
    # Initialize with research-focused components
    asi = ASI(
        config=config,
        mcts_engine=MCTSEngine(),
        synthetic_data_generator=SyntheticDataGenerator()
    )
    
    # Example research query
    task = """
    Research and analyze the latest developments in:
    1. Artificial General Intelligence (AGI)
    2. Recursive Self-Improvement
    3. Neural-Symbolic Integration
    4. Consciousness in AI Systems
    
    Provide a comprehensive analysis including:
    - Key breakthroughs and innovations
    - Current challenges and limitations
    - Future research directions
    - Potential societal impacts
    
    Synthesize information from academic papers, tech blogs, 
    research labs, and industry reports.
    """
    
    result = await asi.run(
        task=task,
        dataset="ai_research_papers",
        stream=True
    )
    
    print("Research Analysis:")
    print(result.solution)
    print("\nMCTS Exploration Path:")
    print(result.consciousness_flow)
    print("\nSynthetic Data Insights:")
    print(result.metrics)

if __name__ == "__main__":
    asyncio.run(web_researcher())
