import pytest
import random
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
    def __init__(self, performance, complexity, innovation):
        self.metrics = {
            'performance': performance,
            'complexity': complexity,
            'innovation': innovation
        }
    
    def get(self, key, default=None):
        return self.metrics.get(key, default)

@pytest.fixture
def code_model_candidates():
    return [
        CodeModelCandidate(0.8, 0.3, 0.7),
        CodeModelCandidate(0.6, 0.5, 0.4),
        CodeModelCandidate(0.9, 0.2, 0.6)
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
    
    assert participant.total_score == 1.5
    assert participant.get_average_performance() == 0.75

def test_single_elimination_tournament(code_model_candidates, judgement_strategy):
    tournament = SingleEliminationTournament(judgement_strategy)
    
    participants = [
        TournamentParticipant(candidate) for candidate in code_model_candidates
    ]
    
    match_result = tournament.match(participants[0], participants[1])
    
    assert 0 <= match_result <= 1

def test_round_robin_tournament(code_model_candidates, judgement_strategy):
    tournament = RoundRobinTournament(judgement_strategy)
    
    participants = [
        TournamentParticipant(candidate) for candidate in code_model_candidates
    ]
    
    match_result = tournament.match(participants[0], participants[1])
    
    assert 0 <= match_result <= 1

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

def test_scored_participant(code_model_candidates):
    participant = ScoredParticipant(code_model_candidates[0], initial_score=0.5)
    
    assert participant.total_score == 0.5
    
    participant.apply_performance_multiplier(1.5)
    
    assert participant.total_score == 0.75

def test_metric_participant(code_model_candidates):
    participant = MetricParticipant(code_model_candidates[0])
    
    participant.add_metric('custom_metric', 0.9)
    
    assert participant.get_metric('custom_metric') == 0.9
    assert participant.get_metric('non_existent', 0.0) == 0.0

def test_adaptive_participant(code_model_candidates):
    participant = AdaptiveParticipant(code_model_candidates[0])
    
    initial_total_score = participant.total_score
    
    participant.record_performance(0.7)
    participant.record_performance(0.8)
    
    assert participant.total_score > initial_total_score
    
    participant.adjust_learning_rate(0.05)
    assert participant.learning_rate == 0.05
