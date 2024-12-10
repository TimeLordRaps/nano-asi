from typing import TypeVar, Generic, List, Any, Optional
from .base import MCTSState

T = TypeVar('T')

class MCTSNode(Generic[T]):
    def __init__(
        self, 
        state: MCTSState[T], 
        parent: Optional['MCTSNode[T]'] = None, 
        action: Optional[Any] = None
    ):
        self.state = state
        self.parent = parent
        self.action = action
        
        self.children: List[MCTSNode[T]] = []
        self.untried_actions = state.get_legal_actions()
        
        self.visits = 0
        self.value = 0.0
    
    def add_child(self, action: Any, new_state: MCTSState[T]) -> 'MCTSNode[T]':
        """
        Add a child node to the current node.
        
        Args:
            action (Any): The action taken to reach the new state.
            new_state (MCTSState[T]): The new state after taking the action.
        
        Returns:
            MCTSNode[T]: The newly created child node.
        """
        child_node = MCTSNode(
            state=new_state, 
            parent=self, 
            action=action
        )
        
        self.children.append(child_node)
        self.untried_actions.remove(action)
        
        return child_node
    
    def best_child(self, exploration_weight: float = 1.0) -> 'MCTSNode[T]':
        """
        Select the best child node using UCB1 formula.
        
        Args:
            exploration_weight (float, optional): Weight for exploration vs exploitation. Defaults to 1.0.
        
        Returns:
            MCTSNode[T]: The best child node.
        """
        import math
        
        if not self.children:
            raise ValueError("No children to select from")
        
        def ucb1_score(node: MCTSNode[T]) -> float:
            if node.visits == 0:
                return float('inf')
            
            exploitation_score = node.value / node.visits
            exploration_score = math.sqrt(
                2 * math.log(self.visits) / node.visits
            )
            
            return exploitation_score + exploration_weight * exploration_score
        
        return max(self.children, key=ucb1_score)
