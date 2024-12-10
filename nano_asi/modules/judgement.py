from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')

class JudgementCriteria(ABC):
    """
    Abstract base class for defining judgement criteria.
    
    Provides a flexible framework for evaluating and scoring different 
    types of solutions, strategies, or candidates.
    """
    
    @abstractmethod
    def evaluate(self, candidate: Any) -> float:
        """
        Evaluate a candidate based on specific criteria.
        
        Args:
            candidate: The item to be evaluated
        
        Returns:
            A score representing the candidate's performance
        """
        pass
    
    @abstractmethod
    def compare(self, candidate1: Any, candidate2: Any) -> int:
        """
        Compare two candidates based on the defined criteria.
        
        Args:
            candidate1: First candidate to compare
            candidate2: Second candidate to compare
        
        Returns:
            -1 if candidate1 is worse, 0 if equal, 1 if candidate1 is better
        """
        pass

class JudgementStrategy(Generic[T], ABC):
    """
    Abstract base class for implementing different judgement strategies.
    
    Allows for flexible and extensible evaluation of candidates across 
    various domains and problem types.
    """
    
    def __init__(self, criteria: List[JudgementCriteria]):
        """
        Initialize the judgement strategy with evaluation criteria.
        
        Args:
            criteria: List of criteria used for evaluation
        """
        self.criteria = criteria
    
    @abstractmethod
    def aggregate_score(self, candidate: T) -> float:
        """
        Aggregate scores from multiple criteria for a candidate.
        
        Args:
            candidate: The candidate to evaluate
        
        Returns:
            Aggregated performance score
        """
        pass
    
    def rank_candidates(self, candidates: List[T]) -> List[T]:
        """
        Rank candidates based on their aggregated scores.
        
        Args:
            candidates: List of candidates to rank
        
        Returns:
            Candidates sorted from best to worst
        """
        return sorted(
            candidates, 
            key=self.aggregate_score, 
            reverse=True
        )
    
    def select_top_candidates(
        self, 
        candidates: List[T], 
        num_candidates: int = 3
    ) -> List[T]:
        """
        Select top performing candidates.
        
        Args:
            candidates: List of candidates to select from
            num_candidates: Number of top candidates to return
        
        Returns:
            Top performing candidates
        """
        return self.rank_candidates(candidates)[:num_candidates]

class MultiCriteriaJudgement(JudgementStrategy[T]):
    """
    Implements a multi-criteria judgement strategy with weighted scoring.
    """
    
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-criteria judgement with optional weights.
        
        Args:
            criteria: List of judgement criteria
            weights: Optional list of weights for each criterion
        """
        super().__init__(criteria)
        
        # If no weights provided, use equal weighting
        self.weights = (
            weights if weights is not None 
            else [1.0 / len(criteria)] * len(criteria)
        )
        
        # Validate weights match number of criteria
        assert len(self.weights) == len(criteria), \
            "Number of weights must match number of criteria"
    
    def aggregate_score(self, candidate: T) -> float:
        """
        Compute weighted aggregate score across all criteria.
        
        Args:
            candidate: Candidate to evaluate
        
        Returns:
            Weighted aggregate performance score
        """
        scores = [
            criterion.evaluate(candidate) * weight
            for criterion, weight in zip(self.criteria, self.weights)
        ]
        return sum(scores)

class ProbabilisticJudgement(JudgementStrategy[T]):
    """
    Implements a probabilistic judgement strategy that introduces 
    randomness and exploration into candidate selection.
    """
    
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        exploration_rate: float = 0.1
    ):
        """
        Initialize probabilistic judgement strategy.
        
        Args:
            criteria: List of judgement criteria
            exploration_rate: Probability of selecting a non-optimal candidate
        """
        super().__init__(criteria)
        self.exploration_rate = exploration_rate
    
    def aggregate_score(self, candidate: T) -> float:
        """
        Compute aggregate score with potential randomness.
        
        Args:
            candidate: Candidate to evaluate
        
        Returns:
            Performance score with potential exploration bonus
        """
        import random
        
        base_score = super().aggregate_score(candidate)
        
        # Add exploration bonus with small probability
        if random.random() < self.exploration_rate:
            base_score *= random.uniform(1.0, 1.5)
        
        return base_score
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')

class JudgementCriteria(ABC):
    """
    Abstract base class for defining judgement criteria.
    
    Provides a flexible framework for evaluating and scoring different 
    types of solutions, strategies, or candidates.
    """
    
    @abstractmethod
    def evaluate(self, candidate: Any) -> float:
        """
        Evaluate a candidate based on specific criteria.
        
        Args:
            candidate: The item to be evaluated
        
        Returns:
            A score representing the candidate's performance
        """
        pass
    
    @abstractmethod
    def compare(self, candidate1: Any, candidate2: Any) -> int:
        """
        Compare two candidates based on the defined criteria.
        
        Args:
            candidate1: First candidate to compare
            candidate2: Second candidate to compare
        
        Returns:
            -1 if candidate1 is worse, 0 if equal, 1 if candidate1 is better
        """
        pass

class JudgementStrategy(Generic[T], ABC):
    """
    Abstract base class for implementing different judgement strategies.
    
    Allows for flexible and extensible evaluation of candidates across 
    various domains and problem types.
    """
    
    def __init__(self, criteria: List[JudgementCriteria]):
        """
        Initialize the judgement strategy with evaluation criteria.
        
        Args:
            criteria: List of criteria used for evaluation
        """
        self.criteria = criteria
    
    @abstractmethod
    def aggregate_score(self, candidate: T) -> float:
        """
        Aggregate scores from multiple criteria for a candidate.
        
        Args:
            candidate: The candidate to evaluate
        
        Returns:
            Aggregated performance score
        """
        pass
    
    def rank_candidates(self, candidates: List[T]) -> List[T]:
        """
        Rank candidates based on their aggregated scores.
        
        Args:
            candidates: List of candidates to rank
        
        Returns:
            Candidates sorted from best to worst
        """
        return sorted(
            candidates, 
            key=self.aggregate_score, 
            reverse=True
        )
    
    def select_top_candidates(
        self, 
        candidates: List[T], 
        num_candidates: int = 3
    ) -> List[T]:
        """
        Select top performing candidates.
        
        Args:
            candidates: List of candidates to select from
            num_candidates: Number of top candidates to return
        
        Returns:
            Top performing candidates
        """
        return self.rank_candidates(candidates)[:num_candidates]

class MultiCriteriaJudgement(JudgementStrategy[T]):
    """
    Implements a multi-criteria judgement strategy with weighted scoring.
    """
    
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-criteria judgement with optional weights.
        
        Args:
            criteria: List of judgement criteria
            weights: Optional list of weights for each criterion
        """
        super().__init__(criteria)
        
        # If no weights provided, use equal weighting
        self.weights = (
            weights if weights is not None 
            else [1.0 / len(criteria)] * len(criteria)
        )
        
        # Validate weights match number of criteria
        assert len(self.weights) == len(criteria), \
            "Number of weights must match number of criteria"
    
    def aggregate_score(self, candidate: T) -> float:
        """
        Compute weighted aggregate score across all criteria.
        
        Args:
            candidate: Candidate to evaluate
        
        Returns:
            Weighted aggregate performance score
        """
        scores = [
            criterion.evaluate(candidate) * weight
            for criterion, weight in zip(self.criteria, self.weights)
        ]
        return sum(scores)

class ProbabilisticJudgement(JudgementStrategy[T]):
    """
    Implements a probabilistic judgement strategy that introduces 
    randomness and exploration into candidate selection.
    """
    
    def __init__(
        self, 
        criteria: List[JudgementCriteria], 
        exploration_rate: float = 0.1
    ):
        """
        Initialize probabilistic judgement strategy.
        
        Args:
            criteria: List of judgement criteria
            exploration_rate: Probability of selecting a non-optimal candidate
        """
        super().__init__(criteria)
        self.exploration_rate = exploration_rate
    
    def aggregate_score(self, candidate: T) -> float:
        """
        Compute aggregate score with potential randomness.
        
        Args:
            candidate: Candidate to evaluate
        
        Returns:
            Performance score with potential exploration bonus
        """
        import random
        
        base_score = super().aggregate_score(candidate)
        
        # Add exploration bonus with small probability
        if random.random() < self.exploration_rate:
            base_score *= random.uniform(1.0, 1.5)
        
        return base_score
