from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Any

T = TypeVar('T')

class MCTSState(Generic[T], ABC):
    """
    Abstract base class representing a state in Monte Carlo Tree Search.
    
    Provides a generic interface for state representation and manipulation.
    """
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """
        Get a list of legal actions from the current state.
        
        Returns:
            List[Any]: List of possible actions.
        """
        pass
    
    @abstractmethod
    def take_action(self, action: Any) -> 'MCTSState[T]':
        """
        Create a new state by taking a specific action.
        
        Args:
            action (Any): The action to take.
        
        Returns:
            MCTSState[T]: A new state after taking the action.
        """
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the current state is a terminal state.
        
        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        """
        Get the reward for the current state.
        
        Returns:
            float: Numerical reward value.
        """
        pass

class MonteCarloTreeSearch(Generic[T]):
    """
    Base implementation of the Monte Carlo Tree Search algorithm.
    
    Provides a flexible framework for search and decision-making.
    """
    
    def __init__(
        self, 
        exploration_weight: float = 1.0,
        max_iterations: int = 1000,
        max_depth: int = 10
    ):
        """
        Initialize the MCTS algorithm.
        
        Args:
            exploration_weight (float, optional): Weight for exploration vs exploitation. Defaults to 1.0.
            max_iterations (int, optional): Maximum number of search iterations. Defaults to 1000.
            max_depth (int, optional): Maximum search depth. Defaults to 10.
        """
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth
    
    def search(self, initial_state: MCTSState[T]) -> Any:
        """
        Perform Monte Carlo Tree Search to find the best action.
        
        Args:
            initial_state (MCTSState[T]): The starting state for the search.
        
        Returns:
            Any: The best action found by the search.
        """
        # Placeholder implementation
        legal_actions = initial_state.get_legal_actions()
        return legal_actions[0] if legal_actions else None
