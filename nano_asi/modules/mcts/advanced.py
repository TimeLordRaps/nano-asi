from .base import MonteCarloTreeSearch, MCTSState
from .nodes import MCTSNode
from typing import TypeVar, Generic, Any
import random
import math

T = TypeVar('T')

class UCBMonteCarloTreeSearch(MonteCarloTreeSearch[T]):
    def __init__(
        self, 
        exploration_weight: float = 1.0,
        max_iterations: int = 1000,
        max_depth: int = 10,
        ucb_variant: str = 'ucb1'
    ):
        super().__init__(
            exploration_weight, 
            max_iterations, 
            max_depth
        )
        self.ucb_variant = ucb_variant
    
    def search(self, initial_state: MCTSState[T]) -> Any:
        root_node = MCTSNode(initial_state)
        
        for _ in range(self.max_iterations):
            node = self._select_node(root_node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        return root_node.best_child(self.exploration_weight).action
    
    def _select_node(self, root_node: MCTSNode[T]) -> MCTSNode[T]:
        current_node = root_node
        
        while not current_node.state.is_terminal():
            if current_node.untried_actions:
                return self._expand(current_node)
            current_node = current_node.best_child(self.exploration_weight)
        
        return current_node
    
    def _expand(self, node: MCTSNode[T]) -> MCTSNode[T]:
        action = random.choice(node.untried_actions)
        new_state = node.state.take_action(action)
        return node.add_child(action, new_state)
    
    def _simulate(self, state: MCTSState[T]) -> float:
        current_state = state
        
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            if not actions:
                break
            action = random.choice(actions)
            current_state = current_state.take_action(action)
        
        return current_state.get_reward()
    
    def _backpropagate(self, node: MCTSNode[T], reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

class ProbabilisticMCTS(MonteCarloTreeSearch[T]):
    def __init__(
        self, 
        exploration_weight: float = 1.0,
        max_iterations: int = 1000,
        max_depth: int = 10,
        exploration_rate: float = 0.2
    ):
        super().__init__(
            exploration_weight, 
            max_iterations, 
            max_depth
        )
        self.exploration_rate = exploration_rate
    
    def search(self, initial_state: MCTSState[T]) -> Any:
        root_node = MCTSNode(initial_state)
        
        for _ in range(self.max_iterations):
            node = self._select_node(root_node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        return self._choose_action(root_node)
    
    def _select_node(self, root_node: MCTSNode[T]) -> MCTSNode[T]:
        current_node = root_node
        
        while not current_node.state.is_terminal():
            if current_node.untried_actions:
                return self._expand(current_node)
            current_node = self._probabilistic_selection(current_node)
        
        return current_node
    
    def _probabilistic_selection(self, node: MCTSNode[T]) -> MCTSNode[T]:
        if random.random() < self.exploration_rate:
            return random.choice(node.children)
        return node.best_child(self.exploration_weight)
    
    def _expand(self, node: MCTSNode[T]) -> MCTSNode[T]:
        action = random.choice(node.untried_actions)
        new_state = node.state.take_action(action)
        return node.add_child(action, new_state)
    
    def _simulate(self, state: MCTSState[T]) -> float:
        current_state = state
        
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            if not actions:
                break
            action = random.choice(actions)
            current_state = current_state.take_action(action)
        
        return current_state.get_reward()
    
    def _backpropagate(self, node: MCTSNode[T], reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _choose_action(self, root_node: MCTSNode[T]) -> Any:
        # Probabilistic action selection
        total_visits = sum(child.visits for child in root_node.children)
        action_probabilities = [
            child.visits / total_visits for child in root_node.children
        ]
        
        return random.choices(
            [child.action for child in root_node.children], 
            weights=action_probabilities
        )[0]

class AdaptiveMCTS(ProbabilisticMCTS):
    def __init__(
        self, 
        exploration_weight: float = 1.0,
        max_iterations: int = 1000,
        max_depth: int = 10,
        learning_rate: float = 0.1
    ):
        super().__init__(
            exploration_weight, 
            max_iterations, 
            max_depth
        )
        self.learning_rate = learning_rate
        self.iteration_count = 0
    
    def search(self, initial_state: MCTSState[T]) -> Any:
        # Dynamically adjust exploration rate
        self.exploration_rate = max(
            0.01, 
            self.exploration_rate * (1 - self.learning_rate * math.log(self.iteration_count + 1))
        )
        
        self.iteration_count += 1
        
        return super().search(initial_state)
