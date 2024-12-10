from .base import JudgementCriteria
from typing import Any

class PerformanceCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('performance', 0.0)

class ComplexityCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return 1.0 - candidate.get('complexity', 0.0)  # Lower complexity is better

class InnovationCriteria(JudgementCriteria):
    def evaluate(self, candidate: Any) -> float:
        return candidate.get('innovation', 0.0)
