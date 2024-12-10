from .base import JudgementStrategy, JudgementCriteria
from typing import List, Any, Optional

class MultiCriteriaJudgement(JudgementStrategy):
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        weights: Optional[List[float]] = None
    ):
        super().__init__(criteria)
        self.weights = weights or [1.0/len(criteria)] * len(criteria)
    
    def aggregate_score(self, candidate: Any) -> float:
        return sum(
            criteria.evaluate(candidate) * weight 
            for criteria, weight in zip(self.criteria, self.weights)
        )

class WeightedJudgementStrategy(MultiCriteriaJudgement):
    pass
