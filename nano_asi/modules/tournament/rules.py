from typing import TypeVar, Generic, Any
from nano_asi.modules.judgement import JudgementStrategy
import random

T = TypeVar('T')

class SingleEliminationTournament(Generic[T]):
    def __init__(
        self, 
        judgement_strategy: JudgementStrategy[T]
    ):
        self.judgement_strategy = judgement_strategy
    
    def match(self, participant1, participant2) -> float:
        """
        Conduct a single match between two participants.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            float: Score of the match (between 0 and 1)
        """
        score1 = self.judgement_strategy.aggregate_score(participant1.candidate)
        score2 = self.judgement_strategy.aggregate_score(participant2.candidate)
        
        # Normalize and add some randomness
        normalized_score = (score1 / (score1 + score2)) + random.uniform(-0.1, 0.1)
        
        return max(0, min(1, normalized_score))
    
    def run_tournament(self, participants):
        """
        Run a single elimination tournament.
        
        Args:
            participants: List of tournament participants
        
        Returns:
            List of tournament results
        """
        results = []
        while len(participants) > 1:
            next_round = []
            for i in range(0, len(participants), 2):
                if i + 1 < len(participants):
                    match_result = self.match(participants[i], participants[i+1])
                    winner = participants[i] if match_result >= 0.5 else participants[i+1]
                    next_round.append(winner)
                    results.append({
                        'participants': [participants[i], participants[i+1]],
                        'winner': winner,
                        'score': match_result
                    })
            participants = next_round
        
        return results

class RoundRobinTournament(SingleEliminationTournament[T]):
    def __init__(
        self, 
        judgement_strategy: JudgementStrategy[T]
    ):
        super().__init__(judgement_strategy)
    
    def match(self, participant1, participant2) -> float:
        """
        Conduct a round-robin style match with more nuanced scoring.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            float: Score of the match (between 0 and 1)
        """
        base_score = super().match(participant1, participant2)
        
        # Add performance history consideration
        history_factor = (
            participant1.get_average_performance() - 
            participant2.get_average_performance()
        ) * 0.1
        
        return max(0, min(1, base_score + history_factor))
    
    def run_tournament(self, participants):
        """
        Run a round-robin tournament with all participants.
        
        Args:
            participants: List of tournament participants
        
        Returns:
            List of tournament results
        """
        results = []
        for i in range(len(participants)):
            for j in range(i+1, len(participants)):
                match_result = self.match(participants[i], participants[j])
                results.append({
                    'participants': [participants[i], participants[j]],
                    'winner': participants[i] if match_result >= 0.5 else participants[j],
                    'score': match_result
                })
        
        return results

class ProbabilisticTournamentRules(SingleEliminationTournament[T]):
    def __init__(
        self, 
        judgement_strategy: JudgementStrategy[T],
        randomness_factor: float = 0.2
    ):
        super().__init__(judgement_strategy)
        self.randomness_factor = randomness_factor
    
    def match(self, participant1, participant2) -> float:
        """
        Conduct a match with increased probabilistic elements.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            float: Score of the match (between 0 and 1)
        """
        base_score = super().match(participant1, participant2)
        
        # Introduce more significant randomness
        probabilistic_adjustment = random.gauss(0, self.randomness_factor)
        
        return max(0, min(1, base_score + probabilistic_adjustment))
