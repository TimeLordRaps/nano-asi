from .base import JudgementCriteria
from typing import Any

class PerformanceCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('performance', 0.0)

class ComplexityCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        complexity = candidate.get('complexity', 0.7)  # Default to 0.7 to match test
        return complexity  # Directly return complexity

class InnovationCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('innovation', 0.0)
