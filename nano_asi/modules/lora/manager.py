import os
import json
import uuid
import torch
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from unsloth import FastLanguageModel
from nano_asi.modules.lora.config import LoRAConfig
from nano_asi.modules.consciousness.tracker import ConsciousnessTracker

@dataclass
class LoRAMetadata:
    """
    Comprehensive metadata for a LoRA adapter.
    
    Tracks provenance, performance, and computational characteristics.
    """
    id: str
    timestamp: str
    base_model: str
    compute_complexity: float
    performance_score: float
    training_context: Dict[str, Any]
    consciousness_metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    version: str = '1.0'
    status: str = 'active'

class LoRAManager:
    """
    Manages the lifecycle of LoRA adapters with advanced tracking and storage.
    
    Handles:
    - Local storage and versioning
    - Compute-aware LoRA generation
    - Metadata tracking
    - Tournament-based selection
    """
    
    def __init__(
        self, 
        base_model: str = 'unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit',
        storage_dir: Optional[str] = None
    ):
        """
        Initialize LoRA manager with configurable storage and base model.
        
        Args:
            base_model: Unsloth base model for LoRA generation
            storage_dir: Directory to store LoRA adapters and metadata
        """
        self.base_model = base_model
        self.storage_dir = storage_dir or os.path.join(
            os.path.expanduser('~'), '.nano_asi', 'lora_adapters'
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.consciousness_tracker = ConsciousnessTracker()
    
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
