import pytest
import random
from nano_asi.modules.judgement import (
    JudgementCriteria, 
    JudgementStrategy,
    PerformanceCriteria,
    ComplexityCriteria,
    InnovationCriteria,
    MultiCriteriaJudgement,
    WeightedJudgementStrategy,
    ProbabilisticJudgement,
    ExplorationJudgementStrategy
)

class CodeGenerationCandidate:
    def __init__(self, performance, complexity, innovation):
        self.metrics = {
            'performance': performance,
            'complexity': complexity,
            'innovation': innovation
        }
    
    def get(self, key, default=None):
        return self.metrics.get(key, default)

@pytest.fixture
def code_generation_candidates():
    return [
        CodeGenerationCandidate(0.8, 0.3, 0.7),
        CodeGenerationCandidate(0.6, 0.5, 0.4),
        CodeGenerationCandidate(0.9, 0.2, 0.6)
    ]

def test_performance_criteria(code_generation_candidates):
    criteria = PerformanceCriteria()
    
    assert criteria.evaluate(code_generation_candidates[0]) == 0.8
    assert criteria.evaluate(code_generation_candidates[1]) == 0.6
    
    assert criteria.compare(
        code_generation_candidates[0], 
        code_generation_candidates[1]
    ) == 1

def test_complexity_criteria(code_generation_candidates):
    criteria = ComplexityCriteria()
    
    assert criteria.evaluate(code_generation_candidates[0]) == 0.3
    assert criteria.evaluate(code_generation_candidates[1]) == 0.5
    
    assert criteria.compare(
        code_generation_candidates[0], 
        code_generation_candidates[1]
    ) == 1

def test_multi_criteria_judgement(code_generation_candidates):
    criteria = [
        PerformanceCriteria(),
        ComplexityCriteria(),
        InnovationCriteria()
    ]
    
    strategy = MultiCriteriaJudgement(criteria)
    
    scores = [strategy.aggregate_score(candidate) for candidate in code_generation_candidates]
    
    assert len(scores) == 3
    assert all(0 <= score <= 2 for score in scores)

def test_probabilistic_judgement(code_generation_candidates):
    criteria = [
        PerformanceCriteria(),
        ComplexityCriteria(),
        InnovationCriteria()
    ]
    
    strategy = ProbabilisticJudgement(criteria, exploration_rate=0.1)
    
    scores = [strategy.aggregate_score(candidate) for candidate in code_generation_candidates]
    
    assert len(scores) == 3
    assert all(scores[i] >= 0 for i in range(len(scores)))

def test_exploration_judgement_strategy(code_generation_candidates):
    criteria = [
        PerformanceCriteria(),
        ComplexityCriteria(),
        InnovationCriteria()
    ]
    
    strategy = ExplorationJudgementStrategy(criteria)
    
    initial_scores = [strategy.aggregate_score(candidate) for candidate in code_generation_candidates]
    
    # Run multiple iterations to test adaptive behavior
    for _ in range(10):
        new_scores = [strategy.aggregate_score(candidate) for candidate in code_generation_candidates]
        assert len(new_scores) == 3
    
    assert strategy.exploration_rate <= 1.0
    assert strategy.exploration_rate >= 0.01
