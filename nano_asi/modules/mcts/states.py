from typing import TypeVar, Generic, List, Any
from .base import MCTSState

T = TypeVar('T')

class DefaultMCTSState(MCTSState[T]):
    """
    A default implementation of MCTSState for basic use cases.
    
    Provides a simple implementation that can be extended or used as a base.
    """
    
    def __init__(
        self, 
        actions: List[Any], 
        max_depth: int = 10
    ):
        """
        Initialize the state with possible actions and max depth.
        
        Args:
            actions (List[Any]): List of possible actions.
            max_depth (int, optional): Maximum search depth. Defaults to 10.
        """
        self._actions = actions
        self._current_depth = 0
        self._max_depth = max_depth
    
    def get_legal_actions(self) -> List[Any]:
        """
        Get a list of legal actions from the current state.
        
        Returns:
            List[Any]: List of possible actions.
        """
        return self._actions if self._current_depth < self._max_depth else []
    
    def take_action(self, action: Any) -> 'DefaultMCTSState[T]':
        """
        Create a new state by taking a specific action.
        
        Args:
            action (Any): The action to take.
        
        Returns:
            DefaultMCTSState[T]: A new state after taking the action.
        """
        new_state = DefaultMCTSState(
            actions=self._actions, 
            max_depth=self._max_depth
        )
        new_state._current_depth = self._current_depth + 1
        return new_state
    
    def is_terminal(self) -> bool:
        """
        Check if the current state is a terminal state.
        
        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return self._current_depth >= self._max_depth
    
    def get_reward(self) -> float:
        """
        Get the reward for the current state.
        
        Returns:
            float: Numerical reward value.
        """
        import random
        return random.uniform(0, 1)
