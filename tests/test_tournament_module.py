import pytest
import random
import numpy as np
from typing import List, Dict, Any
from nano_asi.modules.tournament import (
    TournamentParticipant,
    SingleEliminationTournament,
    RoundRobinTournament,
    ProbabilisticTournamentRules,
    ScoredParticipant,
    MetricParticipant,
    AdaptiveParticipant
)
from nano_asi.modules.judgement import (
    MultiCriteriaJudgement,
    PerformanceCriteria,
    ComplexityCriteria,
    InnovationCriteria
)

class CodeModelCandidate:
    def __init__(
        self, 
        performance: float, 
        complexity: float, 
        innovation: float,
        additional_metrics: Dict[str, float] = None
    ):
        self.metrics = {
            'performance': performance,
            'complexity': complexity,
            'innovation': innovation,
            **(additional_metrics or {})
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)
    
    def compute_aggregate_score(self) -> float:
        """
        Compute an aggregate score across multiple metrics.
        
        Returns:
            Weighted aggregate score
        """
        weights = {
            'performance': 0.4,
            'complexity': 0.3,
            'innovation': 0.3
        }
        
        return sum(
            self.metrics.get(metric, 0) * weights.get(metric, 0)
            for metric in weights
        )

@pytest.fixture
def code_model_candidates() -> List[CodeModelCandidate]:
    return [
        CodeModelCandidate(0.8, 0.3, 0.7),
        CodeModelCandidate(0.6, 0.5, 0.4),
        CodeModelCandidate(0.9, 0.2, 0.6, {
            'scalability': 0.85,
            'robustness': 0.75
        })
    ]

@pytest.fixture
def judgement_strategy(code_model_candidates):
    criteria = [
        PerformanceCriteria(),
        ComplexityCriteria(),
        InnovationCriteria()
    ]
    return MultiCriteriaJudgement(criteria)

def test_tournament_participant(code_model_candidates):
    participant = TournamentParticipant(code_model_candidates[0])
    
    participant.record_performance(0.7)
    participant.record_performance(0.8)
    
    assert participant.total_score == pytest.approx(1.5)
    assert participant.get_average_performance() == pytest.approx(0.75)

def test_single_elimination_tournament(code_model_candidates, judgement_strategy):
    tournament = SingleEliminationTournament(judgement_strategy)
    
    participants = [
        TournamentParticipant(candidate) for candidate in code_model_candidates
    ]
    
    match_result = tournament.match(participants[0], participants[1])
    
    assert 0 <= match_result <= 1
    
    # Test tournament progression
    tournament_results = tournament.run_tournament(participants)
    assert len(tournament_results) > 0

def test_round_robin_tournament(code_model_candidates, judgement_strategy):
    tournament = RoundRobinTournament(judgement_strategy)
    
    participants = [
        TournamentParticipant(candidate) for candidate in code_model_candidates
    ]
    
    match_result = tournament.match(participants[0], participants[1])
    
    assert 0 <= match_result <= 1
    
    # Test full tournament results
    tournament_results = tournament.run_tournament(participants)
    assert len(tournament_results) == len(participants)

def test_probabilistic_tournament(code_model_candidates, judgement_strategy):
    tournament = ProbabilisticTournamentRules(
        judgement_strategy, 
        randomness_factor=0.2
    )
    
    participants = [
        TournamentParticipant(candidate) for candidate in code_model_candidates
    ]
    
    match_result = tournament.match(participants[0], participants[1])
    
    assert 0 <= match_result <= 1
    
    # Test probabilistic variations
    results = [tournament.match(participants[0], participants[1]) for _ in range(10)]
    assert len(set(results)) > 1  # Ensure some variation

def test_scored_participant(code_model_candidates):
    participant = ScoredParticipant(code_model_candidates[0], initial_score=0.5)
    
    assert participant.total_score == pytest.approx(0.5)
    
    participant.apply_performance_multiplier(1.5)
    
    assert participant.total_score == pytest.approx(0.75)

def test_metric_participant(code_model_candidates):
    participant = MetricParticipant(code_model_candidates[0])
    
    participant.add_metric('custom_metric', 0.9)
    participant.add_metric('scalability', 0.85)
    
    assert participant.get_metric('custom_metric') == pytest.approx(0.9)
    assert participant.get_metric('scalability') == pytest.approx(0.85)
    assert participant.get_metric('non_existent', 0.0) == 0.0

def test_adaptive_participant(code_model_candidates):
    participant = AdaptiveParticipant(code_model_candidates[0])
    
    initial_total_score = participant.total_score
    
    participant.record_performance(0.7)
    participant.record_performance(0.8)
    
    assert participant.total_score > initial_total_score
    
    participant.adjust_learning_rate(0.05)
    assert participant.learning_rate == pytest.approx(0.05)

def test_candidate_aggregate_scoring(code_model_candidates):
    candidate = code_model_candidates[2]
    
    aggregate_score = candidate.compute_aggregate_score()
    
    assert 0 <= aggregate_score <= 1
    assert aggregate_score == pytest.approx(
        0.4 * 0.9 + 0.3 * 0.2 + 0.3 * 0.6, 
        rel=1e-2
    )
