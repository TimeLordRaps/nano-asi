from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Any, Optional
from nano_asi.modules.judgement import JudgementStrategy, JudgementCriteria

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
            candidate: The candidate representing the participant
        """
        self.candidate = candidate
        self.performance_history: List[float] = []
        self.total_score: float = 0.0
    
    def record_performance(self, score: float):
        """
        Record performance in the tournament.
        
        Args:
            score: Performance score for a match
        """
        self.performance_history.append(score)
        self.total_score += score

class TournamentRules(Generic[T], ABC):
    """
    Abstract base class defining tournament match dynamics and evaluation rules.
    
    Provides a flexible framework for different tournament structures and scoring.
    """
    
    def __init__(
        self, 
        judgement_strategy: Optional[JudgementStrategy[T]] = None
    ):
        """
        Initialize tournament rules with optional judgement strategy.
        
        Args:
            judgement_strategy: Strategy for evaluating match outcomes
        """
        self.judgement_strategy = judgement_strategy
    
    @abstractmethod
    def match(
        self, 
        participant1: TournamentParticipant[T], 
        participant2: TournamentParticipant[T]
    ) -> float:
        """
        Conduct a match between two participants.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            Performance score for the match
        """
        pass
    
    def tournament(
        self, 
        participants: List[TournamentParticipant[T]], 
        num_rounds: int = 3
    ) -> List[TournamentParticipant[T]]:
        """
        Conduct a tournament with multiple rounds.
        
        Args:
            participants: List of tournament participants
            num_rounds: Number of tournament rounds
        
        Returns:
            Ranked list of participants
        """
        for _ in range(num_rounds):
            # Pair participants for matches
            for i in range(0, len(participants), 2):
                if i + 1 < len(participants):
                    score = self.match(participants[i], participants[i+1])
                    participants[i].record_performance(score)
                    participants[i+1].record_performance(1.0 - score)
        
        # Rank participants by total score
        return sorted(
            participants, 
            key=lambda p: p.total_score, 
            reverse=True
        )

class SingleEliminationTournament(TournamentRules[T]):
    """
    Implements a single elimination tournament structure.
    
    Participants are eliminated after losing a match.
    """
    
    def match(
        self, 
        participant1: TournamentParticipant[T], 
        participant2: TournamentParticipant[T]
    ) -> float:
        """
        Conduct a single elimination match.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            Performance score (1.0 for winner, 0.0 for loser)
        """
        if self.judgement_strategy:
            score1 = self.judgement_strategy.aggregate_score(participant1.candidate)
            score2 = self.judgement_strategy.aggregate_score(participant2.candidate)
            return 1.0 if score1 > score2 else 0.0
        
        # Default random selection if no strategy provided
        import random
        return 1.0 if random.random() > 0.5 else 0.0

class RoundRobinTournament(TournamentRules[T]):
    """
    Implements a round-robin tournament structure.
    
    Each participant plays against every other participant.
    """
    
    def match(
        self, 
        participant1: TournamentParticipant[T], 
        participant2: TournamentParticipant[T]
    ) -> float:
        """
        Conduct a round-robin match.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            Performance score based on relative performance
        """
        if self.judgement_strategy:
            score1 = self.judgement_strategy.aggregate_score(participant1.candidate)
            score2 = self.judgement_strategy.aggregate_score(participant2.candidate)
            
            # Normalize score to range [0, 1]
            total_score = score1 + score2
            return score1 / total_score if total_score > 0 else 0.5
        
        # Default random selection if no strategy provided
        import random
        return random.random()

class ProbabilisticTournamentRules(TournamentRules[T]):
    """
    Implements tournament rules with probabilistic match outcomes.
    
    Introduces controlled randomness into tournament dynamics.
    """
    
    def __init__(
        self, 
        judgement_strategy: Optional[JudgementStrategy[T]] = None,
        randomness_factor: float = 0.2
    ):
        """
        Initialize probabilistic tournament rules.
        
        Args:
            judgement_strategy: Strategy for evaluating match outcomes
            randomness_factor: Level of randomness in match outcomes
        """
        super().__init__(judgement_strategy)
        self.randomness_factor = randomness_factor
    
    def match(
        self, 
        participant1: TournamentParticipant[T], 
        participant2: TournamentParticipant[T]
    ) -> float:
        """
        Conduct a probabilistic match.
        
        Args:
            participant1: First tournament participant
            participant2: Second tournament participant
        
        Returns:
            Performance score with probabilistic variation
        """
        import random
        
        if self.judgement_strategy:
            score1 = self.judgement_strategy.aggregate_score(participant1.candidate)
            score2 = self.judgement_strategy.aggregate_score(participant2.candidate)
            
            # Introduce controlled randomness
            random_adjustment = random.uniform(-self.randomness_factor, self.randomness_factor)
            normalized_score = (score1 / (score1 + score2)) + random_adjustment
            
            return max(0.0, min(1.0, normalized_score))
        
        # Fallback to pure random selection
        return random.random()
