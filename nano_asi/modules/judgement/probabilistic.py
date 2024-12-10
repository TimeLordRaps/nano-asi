from .base import JudgementStrategy, JudgementCriteria
from typing import List, Any
import random

class ProbabilisticJudgement(JudgementStrategy):
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        exploration_rate: float = 0.1
    ):
        super().__init__(criteria)
        self.exploration_rate = exploration_rate
    
    def aggregate_score(self, candidate: Any) -> float:
        base_score = sum(criteria.evaluate(candidate) for criteria in self.criteria)
        
        # Add probabilistic exploration
        exploration_bonus = random.uniform(0, self.exploration_rate * base_score)
        
        return base_score + exploration_bonus

class ExplorationJudgementStrategy(ProbabilisticJudgement):
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        initial_exploration_rate: float = 0.2
    ):
        super().__init__(criteria, initial_exploration_rate)
        self.iteration_count = 0
    
    def aggregate_score(self, candidate: Any) -> float:
        # Gradually reduce exploration rate
        self.exploration_rate = max(
            0.01, 
            self.exploration_rate * (0.99 ** self.iteration_count)
        )
        
        self.iteration_count += 1
        
        return super().aggregate_score(candidate)
