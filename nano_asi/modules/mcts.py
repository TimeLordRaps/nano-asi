from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
import math
import random
import numpy as np

T = TypeVar('T')

class MCTSState(Generic[T], ABC):
    """
    Abstract base class representing a state in Monte Carlo Tree Search.
    
    Provides a generic interface for state representation and manipulation.
    """
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """
        Retrieve all legal actions for the current state.
        
        Returns:
            List of possible actions
        """
        pass
    
    @abstractmethod
    def take_action(self, action: Any) -> 'MCTSState[T]':
        """
        Apply an action to the current state.
        
        Args:
            action: Action to be applied
        
        Returns:
            New state after applying the action
        """
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the current state is a terminal state.
        
        Returns:
            Boolean indicating terminal state
        """
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        """
        Compute the reward for the current state.
        
        Returns:
            Reward value
        """
        pass

class MCTSNode(Generic[T]):
    """
    Represents a node in the Monte Carlo Tree Search algorithm.
    
    Tracks state information, visit counts, and search metrics.
    """
    
    def __init__(
        self, 
        state: MCTSState[T], 
        parent: Optional['MCTSNode[T]'] = None, 
        action: Optional[Any] = None
    ):
        """
        Initialize an MCTS node.
        
        Args:
            state: Current state of the node
            parent: Parent node in the search tree
            action: Action that led to this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        
        self.children: Dict[Any, 'MCTSNode[T]'] = {}
        self.visits = 0
        self.value = 0.0
        
        # Unexplored actions
        self.untried_actions = state.get_legal_actions()
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all actions have been explored.
        
        Returns:
            Boolean indicating full expansion
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float = 1.0) -> 'MCTSNode[T]':
        """
        Select the best child node using UCB1 formula.
        
        Args:
            exploration_weight: Controls exploration vs exploitation
        
        Returns:
            Best child node
        """
        return max(
            self.children.values(),
            key=lambda child: child.value / child.visits + 
                              exploration_weight * math.sqrt(
                                  math.log(self.visits) / child.visits
                              )
        )

class MonteCarloTreeSearch(Generic[T]):
    """
    Implements the Monte Carlo Tree Search algorithm.
    
    Provides a flexible framework for search and decision-making.
    """
    
    def __init__(
        self, 
        exploration_weight: float = 1.0,
        max_iterations: int = 1000,
        max_depth: int = 10
    ):
        """
        Initialize MCTS with configurable parameters.
        
        Args:
            exploration_weight: Controls exploration vs exploitation
            max_iterations: Maximum search iterations
            max_depth: Maximum search depth
        """
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth
    
    def search(self, initial_state: MCTSState[T]) -> Any:
        """
        Perform Monte Carlo Tree Search.
        
        Args:
            initial_state: Starting state for the search
        
        Returns:
            Best action found
        """
        root = MCTSNode(initial_state)
        
        for _ in range(self.max_iterations):
            node = self._select_node(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        
        return root.best_child(0).action
    
    def _select_node(self, root: MCTSNode[T]) -> MCTSNode[T]:
        """
        Select a node for expansion using tree policy.
        
        Args:
            root: Root node of the search tree
        
        Returns:
            Selected node for expansion
        """
        current_node = root
        depth = 0
        
        while not current_node.state.is_terminal() and depth < self.max_depth:
            if not current_node.is_fully_expanded():
                return self._expand(current_node)
            else:
                current_node = current_node.best_child(self.exploration_weight)
                depth += 1
        
        return current_node
    
    def _expand(self, node: MCTSNode[T]) -> MCTSNode[T]:
        """
        Expand a node by selecting an untried action.
        
        Args:
            node: Node to expand
        
        Returns:
            Newly created child node
        """
        action = node.untried_actions.pop()
        new_state = node.state.take_action(action)
        child_node = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child_node
        
        return child_node
    
    def _simulate(self, node: MCTSNode[T]) -> float:
        """
        Perform a random simulation from the given node.
        
        Args:
            node: Starting node for simulation
        
        Returns:
            Reward obtained from simulation
        """
        current_state = node.state
        depth = 0
        
        while not current_state.is_terminal() and depth < self.max_depth:
            actions = current_state.get_legal_actions()
            action = random.choice(actions)
            current_state = current_state.take_action(action)
            depth += 1
        
        return current_state.get_reward()
    
    def _backpropagate(self, node: MCTSNode[T], reward: float):
        """
        Backpropagate the reward through the search tree.
        
        Args:
            node: Starting node for backpropagation
            reward: Reward to propagate
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
