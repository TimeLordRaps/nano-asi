import os
import json
import uuid
import torch
import shutil
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

from unsloth import FastLanguageModel
from nano_asi.modules.lora.config import LoRAConfig
from nano_asi.modules.consciousness.tracker import ConsciousnessTracker
from nano_asi.modules.evaluation.benchmarks import EvaluationSuite
from nano_asi.modules.mcts import MonteCarloTreeSearch
from nano_asi.modules.tournament import TournamentSelection
from nano_asi.modules.storage import ModelVersionController

@dataclass
class LoRATrainingTrajectory:
    """
    Represents the complete training trajectory of a LoRA adapter.
    
    Tracks the evolution, performance, and decision-making process 
    throughout the training lifecycle.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_model_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    training_stages: List[Dict[str, Any]] = field(default_factory=list)
    performance_graph: Dict[str, Any] = field(default_factory=dict)
    pruning_metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_tags: List[str] = field(default_factory=list)

@dataclass
class LoRAMetadata:
    """
    Comprehensive metadata for LoRA adapters with advanced tracking.
    
    Captures the entire lifecycle, evolution, and semantic versioning 
    of a LoRA adapter.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    base_model: str = ''
    compute_complexity: float = 0.0
    performance_score: float = 0.0
    training_context: Dict[str, Any] = field(default_factory=dict)
    consciousness_metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    version: str = '0.1.0'
    status: str = 'active'
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    mcts_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    diffusion_metadata: Optional[Dict[str, Any]] = None
    training_trajectory: Optional[LoRATrainingTrajectory] = None
    semantic_tags: List[str] = field(default_factory=list)

class LoRAManager:
    """
    Advanced LoRA management system with:
    - Dynamic compute-aware LoRA generation
    - Iterative DPO training
    - MCTS-driven sample selection
    - Comprehensive metadata tracking
    - Semantic versioning and model evolution tracking
    """
    
    def __init__(
        self, 
        base_model: str = 'unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit',
        storage_dir: Optional[str] = None,
        evaluation_suite: Optional[EvaluationSuite] = None,
        version_controller: Optional[ModelVersionController] = None
    ):
        """
        Initialize LoRA manager with advanced configuration.
        
        Args:
            base_model: Base model for LoRA generation
            storage_dir: Directory to store LoRA adapters
            evaluation_suite: Optional custom evaluation suite
            version_controller: Optional model version tracking system
        """
        self.base_model = base_model.split('*')[0].strip()
        self.storage_dir = storage_dir or os.path.join(
            os.path.expanduser('~'), '.nano_asi', 'lora_adapters'
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.consciousness_tracker = ConsciousnessTracker()
        self.evaluation_suite = evaluation_suite or EvaluationSuite()
        
        # MCTS configuration for training sample selection
        self.mcts = MonteCarloTreeSearch(
            exploration_weight=1.0,
            max_iterations=100,
            max_depth=5
        )
        
        # Tournament selection for model comparison
        self.tournament = TournamentSelection()
        
        # Model version and trajectory tracking
        self.version_controller = version_controller or ModelVersionController(
            storage_dir=self.storage_dir
        )
    
    def _compute_complexity_score(
        self, 
        training_data: List[Dict[str, Any]]
    ) -> float:
        """
        Dynamically compute problem complexity based on training data.
        
        Args:
            training_data: Dataset used for LoRA training
        
        Returns:
            Complexity score between 0 and 1
        """
        # Complexity based on:
        # 1. Dataset size
        # 2. Variance in data
        # 3. Entropy of tokens
        
        complexity = min(1.0, len(training_data) / 1000)
        return complexity
    
    def generate_lora(
        self, 
        training_data: List[Dict[str, Any]],
        config: Optional[LoRAConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate a LoRA adapter with dynamic complexity-aware configuration.
        
        Args:
            training_data: Dataset for LoRA training
            config: Optional custom LoRA configuration
        
        Returns:
            Generated LoRA adapter with metadata
        """
        complexity = self._compute_complexity_score(training_data)
        
        # Dynamically adjust LoRA hyperparameters based on complexity
        lora_config = config or LoRAConfig(
            lora_r=int(16 * (1 + complexity)),  # Larger rank for complex problems
            lora_alpha=int(64 * (1 + complexity)),
            lora_dropout=min(0.1, complexity * 0.2)
        )
        
        # Use Unsloth for LoRA generation
        # Remove any potential wildcard or invalid characters from model name
        base_model = self.base_model.split('*')[0].strip()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            load_in_4bit=True
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.lora_r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout
        )
        
        # Track consciousness during LoRA generation
        state_data = {
            'training_data': training_data,
            'lora_config': asdict(lora_config)
        }
        consciousness_state = self.consciousness_tracker.track_consciousness(state_data)
        
        # Prepare LoRA metadata
        metadata = LoRAMetadata(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            base_model=self.base_model,
            compute_complexity=complexity,
            performance_score=0.0,  # To be updated after training
            training_context=state_data,
            consciousness_metrics=consciousness_state,
            hyperparameters=asdict(lora_config)
        )
        
        # Save LoRA and metadata
        lora_path = os.path.join(self.storage_dir, metadata.id)
        os.makedirs(lora_path, exist_ok=True)
        
        model.save_pretrained(lora_path)
        
        with open(os.path.join(lora_path, 'metadata.json'), 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return {
            'lora_path': lora_path,
            'metadata': asdict(metadata),
            'model': model,
            'tokenizer': tokenizer
        }
    
    def tournament_selection(
        self, 
        lora_candidates: List[Dict[str, Any]], 
        num_winners: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tournament-based MCTS selection of best LoRA adapters.
        
        Args:
            lora_candidates: List of LoRA adapters
            num_winners: Number of top performers to select
        
        Returns:
            Selected top-performing LoRA adapters
        """
        # Sort candidates by complexity and performance
        sorted_candidates = sorted(
            lora_candidates, 
            key=lambda x: (
                x['metadata']['compute_complexity'], 
                x['metadata']['performance_score']
            ), 
            reverse=True
        )
        
        return sorted_candidates[:num_winners]
    
    def load_lora(self, lora_id: str) -> Dict[str, Any]:
        """
        Load a previously saved LoRA adapter.
        
        Args:
            lora_id: Unique identifier of the LoRA
        
        Returns:
            Loaded LoRA adapter details
        """
        lora_path = os.path.join(self.storage_dir, lora_id)
        
        with open(os.path.join(lora_path, 'metadata.json'), 'r') as f:
            metadata = LoRAMetadata(**json.load(f))
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=2048,
            load_in_4bit=True
        )
        
        return {
            'lora_path': lora_path,
            'metadata': asdict(metadata),
            'model': model,
            'tokenizer': tokenizer
        }
    
    def update_performance(self, lora_id: str, performance_score: float):
        """
        Update the performance score of a LoRA adapter.
        
        Args:
            lora_id: Unique identifier of the LoRA
            performance_score: New performance metric
        """
        lora_path = os.path.join(self.storage_dir, lora_id)
        metadata_path = os.path.join(lora_path, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['performance_score'] = performance_score
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
import os
import json
import uuid
import torch
import shutil
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

from unsloth import FastLanguageModel
from nano_asi.modules.lora.config import LoRAConfig
from nano_asi.modules.consciousness.tracker import ConsciousnessTracker
from nano_asi.modules.evaluation.benchmarks import EvaluationSuite
from nano_asi.modules.mcts import MonteCarloTreeSearch

@dataclass
class LoRAMetadata:
    """
    Comprehensive metadata for a LoRA adapter with advanced tracking.
    
    Tracks provenance, performance, computational characteristics, 
    and training context with semantic versioning.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    base_model: str = ''
    compute_complexity: float = 0.0
    performance_score: float = 0.0
    training_context: Dict[str, Any] = field(default_factory=dict)
    consciousness_metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    version: str = '0.1.0'
    status: str = 'active'
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    mcts_trajectory: List[Dict[str, Any]] = field(default_factory=list)

class LoRAManager:
    """
    Advanced LoRA management system with:
    - Compute-aware generation
    - Tournament-MCTS iterative training
    - Comprehensive metadata tracking
    - Versioned storage
    """
    
    def __init__(
        self, 
        base_model: str = 'unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit',
        storage_dir: Optional[str] = None,
        evaluation_suite: Optional[EvaluationSuite] = None
    ):
        """
        Initialize LoRA manager with advanced configuration.
        
        Args:
            base_model: Base model for LoRA generation
            storage_dir: Directory to store LoRA adapters and metadata
            evaluation_suite: Optional custom evaluation suite
        """
        self.base_model = base_model.split('*')[0].strip()
        self.storage_dir = storage_dir or os.path.join(
            os.path.expanduser('~'), '.nano_asi', 'lora_adapters'
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.consciousness_tracker = ConsciousnessTracker()
        self.evaluation_suite = evaluation_suite or EvaluationSuite()
        
        # MCTS configuration for training sample selection
        self.mcts = MonteCarloTreeSearch(
            exploration_weight=1.0,
            max_iterations=100,
            max_depth=5
        )
    
    def _compute_complexity_score(
        self, 
        training_data: List[Dict[str, Any]],
        compute_resources: Dict[str, Any]
    ) -> float:
        """
        Dynamically compute problem complexity with multi-factor analysis.
        
        Args:
            training_data: Dataset for complexity assessment
            compute_resources: Available computational resources
        
        Returns:
            Complexity score between 0 and 1
        """
        # Complexity factors:
        # 1. Dataset characteristics
        # 2. Computational resources
        # 3. Model size requirements
        
        data_complexity = min(1.0, len(training_data) / 1000)
        gpu_memory = compute_resources.get('gpu_memory', 4)  # Default 4GB
        compute_factor = min(1.0, gpu_memory / 16)  # Normalize against 16GB
        
        return np.mean([data_complexity, compute_factor])
    
    def generate_lora(
        self, 
        training_data: List[Dict[str, Any]],
        compute_resources: Optional[Dict[str, Any]] = None,
        config: Optional[LoRAConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate a LoRA adapter with dynamic complexity-aware configuration.
        
        Args:
            training_data: Dataset for LoRA training
            compute_resources: Available computational resources
            config: Optional custom LoRA configuration
        
        Returns:
            Generated LoRA adapter with comprehensive metadata
        """
        compute_resources = compute_resources or {
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        
        complexity = self._compute_complexity_score(training_data, compute_resources)
        
        # Dynamically adjust LoRA hyperparameters based on complexity
        lora_config = config or LoRAConfig(
            lora_r=int(16 * (1 + complexity)),  # Larger rank for complex problems
            lora_alpha=int(64 * (1 + complexity)),
            lora_dropout=min(0.1, complexity * 0.2),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        
        # Use Unsloth for LoRA generation
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=2048,
            load_in_4bit=True
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.lora_r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout
        )
        
        # Evaluate model on benchmark suite
        evaluation_results = self.evaluation_suite.evaluate(model, training_data)
        
        # Prepare LoRA metadata
        metadata = LoRAMetadata(
            base_model=self.base_model,
            compute_complexity=complexity,
            performance_score=evaluation_results.get('overall_score', 0.0),
            training_context={'training_data_size': len(training_data)},
            hyperparameters=asdict(lora_config),
            evaluation_results=evaluation_results
        )
        
        # Save LoRA and metadata
        lora_path = os.path.join(self.storage_dir, metadata.id)
        os.makedirs(lora_path, exist_ok=True)
        
        model.save_pretrained(lora_path)
        
        with open(os.path.join(lora_path, 'metadata.json'), 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return {
            'lora_path': lora_path,
            'metadata': asdict(metadata),
            'model': model,
            'tokenizer': tokenizer
        }
    
    def tournament_selection(
        self, 
        lora_candidates: List[Dict[str, Any]], 
        num_winners: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tournament-based MCTS selection of best LoRA adapters.
        
        Args:
            lora_candidates: List of LoRA adapters
            num_winners: Number of top performers to select
        
        Returns:
            Selected top-performing LoRA adapters
        """
        # Implement MCTS-driven selection
        selected_candidates = self.mcts.select_best_candidates(
            lora_candidates, 
            evaluation_metric='performance_score',
            num_winners=num_winners
        )
        
        return selected_candidates
import os
import json
import uuid
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

from unsloth import FastLanguageModel
from nano_asi.modules.lora.config import LoRAConfig
from nano_asi.modules.consciousness.tracker import ConsciousnessTracker
from nano_asi.modules.evaluation.benchmarks import EvaluationSuite
from nano_asi.modules.mcts import MonteCarloTreeSearch
from nano_asi.modules.tournament import TournamentSelection
from nano_asi.modules.storage import ModelVersionController

@dataclass
class LoRATrainingTrajectory:
    """
    Represents the complete training trajectory of a LoRA adapter.
    
    Tracks the evolution, performance, and decision-making process 
    throughout the training lifecycle.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_model_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    training_stages: List[Dict[str, Any]] = field(default_factory=list)
    performance_graph: Dict[str, Any] = field(default_factory=dict)
    pruning_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoRAMetadata:
    """
    Comprehensive metadata for LoRA adapters with advanced tracking.
    
    Captures the entire lifecycle, evolution, and semantic versioning 
    of a LoRA adapter.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    base_model: str = ''
    compute_complexity: float = 0.0
    performance_score: float = 0.0
    training_context: Dict[str, Any] = field(default_factory=dict)
    consciousness_metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    version: str = '0.1.0'
    status: str = 'active'
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    mcts_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    diffusion_metadata: Optional[Dict[str, Any]] = None
    training_trajectory: Optional[LoRATrainingTrajectory] = None

class LoRAManager:
    """
    Advanced LoRA management system with:
    - Dynamic compute-aware LoRA generation
    - Iterative DPO training
    - MCTS-driven sample selection
    - Comprehensive metadata tracking
    - Semantic versioning and model evolution tracking
    """
    
    def __init__(
        self, 
        base_model: str = 'unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit',
        storage_dir: Optional[str] = None,
        evaluation_suite: Optional[EvaluationSuite] = None,
        version_controller: Optional[ModelVersionController] = None
    ):
        """
        Initialize LoRA manager with advanced configuration.
        
        Args:
            base_model: Base model for LoRA generation
            storage_dir: Directory to store LoRA adapters
            evaluation_suite: Optional custom evaluation suite
            version_controller: Optional model version tracking system
        """
        self.base_model = base_model.split('*')[0].strip()
        self.storage_dir = storage_dir or os.path.join(
            os.path.expanduser('~'), '.nano_asi', 'lora_adapters'
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.consciousness_tracker = ConsciousnessTracker()
        self.evaluation_suite = evaluation_suite or EvaluationSuite()
        
        # MCTS configuration for training sample selection
        self.mcts = MonteCarloTreeSearch(
            exploration_weight=1.0,
            max_iterations=100,
            max_depth=5
        )
        
        # Tournament selection for model comparison
        self.tournament = TournamentSelection()
        
        # Model version and trajectory tracking
        self.version_controller = version_controller or ModelVersionController(
            storage_dir=self.storage_dir
        )
    
    def _compute_complexity_score(
        self, 
        training_data: List[Dict[str, Any]],
        compute_resources: Dict[str, Any]
    ) -> float:
        """
        Dynamically compute problem complexity with multi-factor analysis.
        
        Factors considered:
        - Dataset characteristics
        - Computational resources
        - Problem domain complexity
        
        Args:
            training_data: Dataset for complexity assessment
            compute_resources: Available computational resources
        
        Returns:
            Complexity score between 0 and 1
        """
        data_complexity = min(1.0, len(training_data) / 1000)
        gpu_memory = compute_resources.get('gpu_memory', 4)  # Default 4GB
        compute_factor = min(1.0, gpu_memory / 16)  # Normalize against 16GB
        
        # Additional complexity factors can be added here
        domain_complexity = self._assess_domain_complexity(training_data)
        
        return np.mean([data_complexity, compute_factor, domain_complexity])
    
    def _assess_domain_complexity(self, training_data: List[Dict[str, Any]]) -> float:
        """
        Assess the complexity of the problem domain.
        
        Args:
            training_data: Dataset to analyze
        
        Returns:
            Domain complexity score
        """
        # Implement domain-specific complexity assessment
        # This could involve analyzing token diversity, semantic complexity, etc.
        return np.random.random()  # Placeholder
    
    def generate_lora(
        self, 
        training_data: List[Dict[str, Any]],
        compute_resources: Optional[Dict[str, Any]] = None,
        config: Optional[LoRAConfig] = None,
        diffusion_model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Generate a LoRA adapter with dynamic complexity-aware configuration.
        
        Args:
            training_data: Dataset for LoRA training
            compute_resources: Available computational resources
            config: Optional custom LoRA configuration
            diffusion_model: Optional diffusion model for LoRA size adjustment
        
        Returns:
            Generated LoRA adapter with comprehensive metadata
        """
        compute_resources = compute_resources or {
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        
        complexity = self._compute_complexity_score(training_data, compute_resources)
        
        # Dynamically adjust LoRA size based on complexity and optional diffusion model
        if diffusion_model:
            lora_size = self._adjust_lora_size_with_diffusion(complexity, diffusion_model)
        else:
            lora_size = self._compute_default_lora_size(complexity)
        
        # Dynamically adjust LoRA hyperparameters based on complexity
        lora_config = config or LoRAConfig(
            lora_r=lora_size['rank'],
            lora_alpha=lora_size['alpha'],
            lora_dropout=min(0.1, complexity * 0.2),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        
        # Use Unsloth for LoRA generation
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=2048,
            load_in_4bit=True
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.lora_r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout
        )
        
        # Evaluate model on benchmark suite
        evaluation_results = self.evaluation_suite.evaluate(model, training_data)
        
        # Prepare LoRA metadata
        metadata = LoRAMetadata(
            base_model=self.base_model,
            compute_complexity=complexity,
            performance_score=evaluation_results.get('overall_score', 0.0),
            training_context={'training_data_size': len(training_data)},
            hyperparameters=asdict(lora_config),
            evaluation_results=evaluation_results,
            diffusion_metadata=lora_size.get('diffusion_metadata')
        )
        
        # Save LoRA and metadata
        lora_path = os.path.join(self.storage_dir, metadata.id)
        os.makedirs(lora_path, exist_ok=True)
        
        model.save_pretrained(lora_path)
        
        with open(os.path.join(lora_path, 'metadata.json'), 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return {
            'lora_path': lora_path,
            'metadata': asdict(metadata),
            'model': model,
            'tokenizer': tokenizer
        }
    
    def _compute_default_lora_size(self, complexity: float) -> Dict[str, Any]:
        """
        Compute default LoRA size based on complexity.
        
        Args:
            complexity: Problem complexity score
        
        Returns:
            Dictionary with LoRA size parameters
        """
        # Implement logic to determine LoRA size based on complexity
        return {
            'rank': int(16 * (1 + complexity)),
            'alpha': int(64 * (1 + complexity))
        }
    
    def _adjust_lora_size_with_diffusion(
        self, 
        complexity: float, 
        diffusion_model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Adjust LoRA size using a diffusion model.
        
        Args:
            complexity: Problem complexity score
            diffusion_model: Diffusion model for size adjustment
        
        Returns:
            Dictionary with LoRA size parameters and diffusion metadata
        """
        # Implement diffusion-based LoRA size adjustment
        # This could involve using the diffusion model to generate optimal LoRA parameters
        base_size = self._compute_default_lora_size(complexity)
        
        # Placeholder for actual diffusion-based adjustment
        diffusion_metadata = {
            'diffusion_steps': 10,
            'noise_schedule': 'linear'
        }
        
        return {
            **base_size,
            'diffusion_metadata': diffusion_metadata
        }
    
    def iterative_dpo_training(
        self, 
        initial_lora: Dict[str, Any], 
        training_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform iterative Direct Preference Optimization (DPO) training.
        
        Args:
            initial_lora: Initial LoRA adapter
            training_data: Training dataset
        
        Returns:
            List of LoRA adapters trained through iterative DPO
        """
        trained_loras = [initial_lora]
        
        for iteration in range(5):  # Configurable number of iterations
            # MCTS-driven sample selection
            selected_samples = self.mcts.select_best_candidates(
                training_data, 
                evaluation_metric='performance',
                num_winners=10
            )
            
            # Train LoRA on selected samples
            new_lora = self.generate_lora(
                training_data=selected_samples,
                config=initial_lora['metadata']['hyperparameters']
            )
            
            # Evaluate and compare with previous iterations
            tournament_results = self.tournament.run(
                candidates=[trained_loras[-1], new_lora],
                evaluation_suite=self.evaluation_suite
            )
            
            # Add best performing LoRA to trained_loras
            trained_loras.append(tournament_results['winner'])
        
        return trained_loras
    
    def tournament_selection(
        self, 
        lora_candidates: List[Dict[str, Any]], 
        num_winners: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Tournament-based selection of best LoRA adapters.
        
        Args:
            lora_candidates: List of LoRA adapters
            num_winners: Number of top performers to select
        
        Returns:
            Selected top-performing LoRA adapters
        """
        return self.tournament.run(
            candidates=lora_candidates,
            num_winners=num_winners,
            evaluation_suite=self.evaluation_suite
        )['top_candidates']
