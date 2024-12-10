from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Any, Optional

T = TypeVar('T')

class TournamentParticipant(Generic[T]):
    """
    Represents a participant in a tournament with evaluation capabilities.
    
    Allows for flexible representation of candidates across different domains.
    """
    
    def __init__(self, candidate: T):
        """
        Initialize a tournament participant.
        
        Args:
            candidate (T): The candidate being evaluated.
        """
        self.candidate = candidate
        self.total_score = 0.0
        self.performance_history: List[float] = []
    
    def record_performance(self, score: float):
        """
        Record a performance score for the participant.
        
        Args:
            score (float): Performance score to record.
        """
        self.performance_history.append(score)
        self.total_score += score
    
    def get_average_performance(self) -> float:
        """
        Calculate the average performance across all recorded scores.
        
        Returns:
            float: Average performance score.
        """
        return (
            sum(self.performance_history) / len(self.performance_history) 
            if self.performance_history 
            else 0.0
        )
