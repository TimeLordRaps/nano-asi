"""Example of using NanoASI as a scientific research assistant."""

import asyncio
from nano_asi import ASI, Config
from nano_asi.modules import UniverseExplorer, JudgmentSystem

async def research_assistant():
    """Run NanoASI as an advanced scientific research assistant."""
    config = Config(
        universe_exploration={
            "num_parallel_universes": 10,
            "exploration_strategy": "adaptive_mcts",
            "cross_universe_coherence_threshold": 0.8
        },
        judgment_hierarchy={
            "levels": 5,
            "criteria": [
                "scientific_rigor",
                "novelty",
                "feasibility",
                "impact_potential",
                "cross_domain_applicability"
            ]
        }
    )
    
    # Initialize with research-focused components
    asi = ASI(
        config=config,
        universe_explorer=UniverseExplorer(),
        judgment_system=JudgmentSystem()
    )
    
    # Example research task
    task = """
    Analyze recent breakthroughs in quantum computing and propose novel 
    research directions that could lead to significant advances in:
    1. Error correction
    2. Qubit coherence
    3. Quantum-classical interfaces
    4. Practical quantum advantage
    Provide detailed research hypotheses and experimental approaches.
    """
    
    result = await asi.run(
        task=task,
        dataset=["quantum_computing_papers", "physics_preprints"],
        stream=True
    )
    
    print("Research Analysis:")
    print(result.solution)
    print("\nUniverse Exploration Results:")
    print(result.universe_explorations)
    print("\nJudgment Analysis:")
    for level, judgments in enumerate(result.consciousness_flow):
        print(f"\nLevel {level} Insights:")
        print(judgments)

if __name__ == "__main__":
    asyncio.run(research_assistant())
