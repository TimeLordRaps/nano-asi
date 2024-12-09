"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from nano-asi.modules import ConsciousnessTracker, LoRAGenerator
from nano-asi.core import Config

@pytest.fixture
def config():
    """Basic configuration fixture."""
    return Config()

@pytest.fixture
def consciousness_tracker():
    """ConsciousnessTracker instance fixture."""
    return ConsciousnessTracker()

@pytest.fixture
def lora_generator():
    """LoRAGenerator instance fixture."""
    return LoRAGenerator()

@pytest.fixture
def sample_conditional_tokens():
    """Sample conditional tokens for LoRA generation."""
    return torch.randn(1, 128, 64)  # Batch x Seq x Hidden

@pytest.fixture
def sample_flow_data() -> Dict[str, Any]:
    """Sample flow data for consciousness optimization."""
    return {
        'hyperparameters': {
            'lora_r': 64,
            'lora_alpha': 64,
            'lora_dropout': 0.0
        },
        'candidate_stats': [
            {
                'params': torch.randn(64, 64).numpy().tolist(),
                'activation_trace': {
                    'layer_stats': {'mean': 0.0, 'std': 1.0},
                    'pattern_type': 'dense_uniform'
                }
            }
        ]
    }

@pytest.fixture
def validation_data():
    """Sample validation data for meta-optimization."""
    return [
        {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128),
            'labels': torch.randint(0, 1000, (1, 128))
        }
        for _ in range(10)
    ]

@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data for testing consciousness tracking with enhanced complexity."""
    return {
        'activations': [
            {
                'layer_type': 'attention',
                'values': torch.randn(10, 10).numpy().tolist(),
                'gradients': torch.randn(10, 10).numpy().tolist(),
                'pattern_type': 'quantum_coherent',
                'meta_context': {
                    'complexity_score': 0.75,
                    'innovation_potential': 0.6
                }
            },
            {
                'layer_type': 'ffn',
                'values': torch.randn(10, 10).numpy().tolist(),
                'gradients': torch.randn(10, 10).numpy().tolist(),
                'pattern_type': 'recursive_adaptive',
                'meta_context': {
                    'complexity_score': 0.85,
                    'innovation_potential': 0.7
                }
            }
        ],
        'thoughts': [
            {
                'content': 'Initial analysis of problem space',
                'dependencies': [],
                'meta_level': 0,
                'cognitive_trajectory': {
                    'exploration_depth': 0.3,
                    'uncertainty_factor': 0.6
                }
            },
            {
                'content': 'Reflection on analysis approach',
                'dependencies': [0],
                'meta_level': 1,
                'cognitive_trajectory': {
                    'exploration_depth': 0.7,
                    'uncertainty_factor': 0.3
                }
            }
        ],
        'universe_context': {
            'parallel_exploration_count': 3,
            'cross_universe_coherence': 0.65
        }
    }
