"""Core modules for the NanoASI framework."""

__all__ = [
    "ConsciousnessTracker",
    "LoRAGenerator", 
    "MCTSEngine",
    "JudgmentSystem",
    "UniverseExplorer",
    "SyntheticDataGenerator",
    "GraphRAGModule"
]

# Use lazy imports to break circular dependency
def ConsciousnessTracker():
    from .consciousness import ConsciousnessTracker
    return ConsciousnessTracker()

def LoRAGenerator():
    from .lora import LoRAGenerator
    return LoRAGenerator()

def MCTSEngine():
    from .mcts import MCTSEngine
    return MCTSEngine()

def JudgmentSystem():
    from .judgment import JudgmentSystem
    return JudgmentSystem()

def UniverseExplorer():
    from .universe import UniverseExplorer
    return UniverseExplorer()

def SyntheticDataGenerator():
    from .synthetic import SyntheticDataGenerator
    return SyntheticDataGenerator()

def GraphRAGModule():
    from .graph_rag import GraphRAGModule
    return GraphRAGModule()
