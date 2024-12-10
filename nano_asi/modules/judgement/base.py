from abc import ABC, abstractmethod
from typing import List, Any, TypeVar, Generic

T = TypeVar('T')

class JudgementCriteria(ABC):
    """
    Abstract base class for defining evaluation criteria.
    
    Provides a standard interface for evaluating and comparing candidates.
    """
    
    @abstractmethod
    def evaluate(self, candidate: Any) -> float:
        """
        Evaluate a candidate based on specific criteria.
        
        Args:
            candidate (Any): The candidate to be evaluated.
        
        Returns:
            float: A numerical score representing the candidate's performance.
        """
        pass
    
    def compare(self, candidate1: Any, candidate2: Any) -> int:
        """
        Compare two candidates based on the criteria.
        
        Args:
            candidate1 (Any): First candidate to compare.
            candidate2 (Any): Second candidate to compare.
        
        Returns:
            int: 1 if candidate1 is better, -1 if candidate2 is better, 0 if equal.
        """
        score1 = self.evaluate(candidate1)
        score2 = self.evaluate(candidate2)
        
        return 1 if score1 > score2 else (-1 if score2 > score1 else 0)

class JudgementStrategy(Generic[T], ABC):
    """
    Abstract base class for implementing judgement strategies.
    
    Provides a flexible framework for evaluating and scoring candidates.
    """
    
    def __init__(self, criteria: List[JudgementCriteria]):
        """
        Initialize the judgement strategy with evaluation criteria.
        
        Args:
            criteria (List[JudgementCriteria]): List of criteria to use for evaluation.
        """
        self.criteria = criteria
    
    @abstractmethod
    def aggregate_score(self, candidate: T) -> float:
        """
        Aggregate scores across multiple criteria for a candidate.
        
        Args:
            candidate (T): The candidate to evaluate.
        
        Returns:
            float: Aggregated score representing the candidate's overall performance.
        """
        pass
