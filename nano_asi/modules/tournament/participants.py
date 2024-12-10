from .base import TournamentParticipant
from typing import Any, Dict

class ScoredParticipant(TournamentParticipant):
    def __init__(self, candidate: Any, initial_score: float = 0.0):
        super().__init__(candidate)
        self.total_score = initial_score
    
    def apply_performance_multiplier(self, multiplier: float):
        """
        Apply a performance multiplier to the total score.
        
        Args:
            multiplier (float): Multiplier to adjust the total score.
        """
        self.total_score *= multiplier

class MetricParticipant(TournamentParticipant):
    def __init__(self, candidate: Any):
        super().__init__(candidate)
        self.custom_metrics: Dict[str, float] = {}
    
    def add_metric(self, metric_name: str, value: float):
        """
        Add a custom metric to the participant.
        
        Args:
            metric_name (str): Name of the metric.
            value (float): Value of the metric.
        """
        self.custom_metrics[metric_name] = value
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        """
        Retrieve a custom metric.
        
        Args:
            metric_name (str): Name of the metric to retrieve.
            default (float, optional): Default value if metric not found.
        
        Returns:
            float: Value of the metric.
        """
        return self.custom_metrics.get(metric_name, default)

class AdaptiveParticipant(TournamentParticipant):
    def __init__(
        self, 
        candidate: Any, 
        initial_learning_rate: float = 0.1
    ):
        super().__init__(candidate)
        self.learning_rate = initial_learning_rate
    
    def record_performance(self, score: float):
        """
        Record performance with adaptive learning.
        
        Args:
            score (float): Performance score to record.
        """
        super().record_performance(score)
        
        # Dynamically adjust total score based on learning rate
        self.total_score += self.learning_rate * score
    
    def adjust_learning_rate(self, new_rate: float):
        """
        Adjust the learning rate.
        
        Args:
            new_rate (float): New learning rate.
        """
        self.learning_rate = max(0.01, min(1.0, new_rate))
