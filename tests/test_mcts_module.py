import pytest
import random
from typing import List, Any
from nano_asi.modules.mcts import (
    MonteCarloTreeSearch,
    MCTSState,
    MCTSNode,
    UCBMonteCarloTreeSearch,
    ProbabilisticMCTS,
    AdaptiveMCTS
)

class CodeGenerationState(MCTSState):
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
    
    def get_legal_actions(self) -> List[Any]:
        return ['refactor', 'optimize', 'comment', 'test'] if self.depth < self.max_depth else []
    
    def take_action(self, action: str) -> 'CodeGenerationState':
        return CodeGenerationState(self.depth + 1, self.max_depth)
    
    def is_terminal(self) -> bool:
        return self.depth >= self.max_depth
    
    def get_reward(self) -> float:
        return random.uniform(0, 1)

def test_mcts_base_implementation():
    initial_state = CodeGenerationState()
    mcts = MonteCarloTreeSearch(max_iterations=100)
    
    best_action = mcts.search(initial_state)
    
    assert best_action in ['refactor', 'optimize', 'comment', 'test']

def test_ucb_monte_carlo_tree_search():
    initial_state = CodeGenerationState()
    mcts = UCBMonteCarloTreeSearch(
        max_iterations=200, 
        ucb_variant='ucb1'
    )
    
    best_action = mcts.search(initial_state)
    
    assert best_action in ['refactor', 'optimize', 'comment', 'test']

def test_probabilistic_mcts():
    initial_state = CodeGenerationState()
    mcts = ProbabilisticMCTS(
        max_iterations=150, 
        exploration_rate=0.2
    )
    
    best_action = mcts.search(initial_state)
    
    assert best_action in ['refactor', 'optimize', 'comment', 'test']

def test_adaptive_mcts():
    initial_state = CodeGenerationState()
    mcts = AdaptiveMCTS(
        max_iterations=250, 
        learning_rate=0.1
    )
    
    best_action = mcts.search(initial_state)
    
    assert best_action in ['refactor', 'optimize', 'comment', 'test']

def test_mcts_node():
    initial_state = CodeGenerationState()
    node = MCTSNode(initial_state)
    
    assert len(node.untried_actions) > 0
    assert node.visits == 0
    assert node.value == 0.0

def test_mcts_node_expansion():
    initial_state = CodeGenerationState()
    node = MCTSNode(initial_state)
    
    action = node.untried_actions[0]
    new_state = initial_state.take_action(action)
    child_node = node.add_child(action, new_state)
    
    assert child_node.parent == node
    assert child_node.action == action
    assert action not in node.untried_actions
