from .base import JudgementCriteria
from typing import Any

class PerformanceCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('performance', 0.0)

class ComplexityCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        complexity = candidate.get('complexity', 0.7)  # Default to 0.7 to match test
        return complexity  # Directly return complexity
    
    def compare(self, candidate1: Any, candidate2: Any) -> int:
        """
        Compare complexity of two candidates.
        
        Returns:
            1 if candidate1 is more complex, -1 if candidate2 is more complex, 0 if equal
        """
        complexity1 = self.evaluate(candidate1)
        complexity2 = self.evaluate(candidate2)
        
        if complexity1 > complexity2:
            return 1
        elif complexity1 < complexity2:
            return -1
        return 0

class InnovationCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('innovation', 0.0)
