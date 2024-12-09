# Configuration Reference

## Overview

The `Config` class provides comprehensive configuration for NanoASI's recursive self-improvement and temporal investment tracking.

## Core Configuration Components

### TokenInvestmentConfig

Controls token processing and temporal investment tracking:

```python
token_investment = TokenInvestmentConfig(
    total_tokens_processed: int = 0,
    temporal_entropy: float = 0.0,
    productivity_multiplier: float = 1.0,
    temporal_efficiency_score: float = 0.0
)
```

### UniverseExplorationConfig

Configures parallel universe exploration:

```python
universe_exploration = UniverseExplorationConfig(
    num_parallel_universes: int = 5,
    exploration_strategy: str = "adaptive_mcts",
    cross_universe_coherence_threshold: float = 0.7
)
```

### Model Configuration

Base model and adaptation settings:

```python
model_config = {
    "base_model": "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
    "lora_rank": 256,
    "lora_alpha": 256,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

## Temporal Metrics

### Token Investment Tracking

```python
class TokenInvestmentConfig:
    """Advanced configuration for tracking token investment as temporal capital."""
    
    # Fundamental token tracking
    total_tokens_processed: int
    temporal_entropy: float
    
    # Productivity metrics
    productivity_multiplier: float
    temporal_efficiency_score: float
    cognitive_complexity_trajectory: List[float]
    
    # Advanced temporal tracking
    iteration_timestamps: List[float]
    token_investment_history: List[Dict[str, Any]]
    
    # Quantum-inspired metrics
    quantum_token_potential: Dict[str, Any]
    temporal_complexity_metrics: Dict[str, Any]
```

### Temporal Analysis Methods

```python
def analyze_temporal_progression(self) -> Dict[str, Any]:
    """Analyze system's temporal progression."""
    return {
        'learning_acceleration': self._compute_learning_acceleration(),
        'complexity_growth_rate': self._compute_complexity_growth_rate(),
        'innovation_trajectory': self._compute_innovation_trajectory()
    }
```

## Customization Options

### Platform Configuration

```python
platform_config = {
    "supported_platforms": ["local", "cloud", "web", "notebook"],
    "cross_platform_compatibility": True,
    "adaptive_resource_allocation": True
}
```

### Hypertraining Configuration

```python
hypertraining_config = {
    "max_boost_cycles": 10,
    "learning_acceleration_factor": 1.5,
    "meta_learning_enabled": True,
    "exploration_decay_rate": 0.9,
    "innovation_threshold": 0.2
}
```

### Synthetic Data Generation

```python
synthetic_data_config = {
    "generation_strategies": [
        "domain_extrapolation",
        "adversarial_generation",
        "meta_learning_augmentation"
    ],
    "diversity_threshold": 0.75,
    "complexity_scaling": True,
    "meta_data_tracking": True
}
```

## Usage Examples

### Basic Configuration

```python
from nano_asi import Config

config = Config(
    universe_exploration={
        "num_parallel_universes": 5,
        "exploration_strategy": "adaptive_mcts"
    }
)
```

### Advanced Configuration

```python
config = Config(
    token_investment=TokenInvestmentConfig(
        temporal_entropy=0.5,
        productivity_multiplier=1.5
    ),
    universe_exploration=UniverseExplorationConfig(
        num_parallel_universes=10,
        cross_universe_coherence_threshold=0.8
    ),
    model_config={
        "lora_rank": 256,
        "lora_alpha": 256
    }
)
```

### Temporal Investment Configuration

```python
config = Config(
    token_investment=TokenInvestmentConfig(
        meta_learning_dynamics={
            "base_learning_rate": 0.01,
            "adaptive_rate_adjustments": [],
            "learning_acceleration_curve": []
        },
        temporal_complexity_metrics={
            "information_density": 0.0,
            "complexity_growth_rate": 0.0,
            "entropy_reduction_rate": 0.0
        }
    )
)
```

## Best Practices

1. **Token Investment Tracking**
   - Monitor temporal efficiency with `temporal_efficiency_score`
   - Track productivity trends with `productivity_multiplier`

2. **Universe Exploration**
   - Adjust `num_parallel_universes` based on task complexity
   - Set appropriate `cross_universe_coherence_threshold`

3. **Model Configuration**
   - Use appropriate `lora_rank` for your task
   - Configure `target_modules` based on model architecture

4. **Temporal Metrics**
   - Monitor `temporal_entropy` for optimization quality
   - Track `cognitive_complexity_trajectory` for learning progress

## Advanced Topics

### Quantum-Inspired Metrics

```python
quantum_metrics = {
    "quantum_token_potential": {
        "base_potential": 0.0,
        "domain_specific_potentials": {},
        "emergence_probability": {},
        "temporal_coherence_score": 0.0
    }
}
```

### Meta-Learning Configuration

```python
meta_learning_config = {
    "improvement_threshold": 0.1,
    "recursive_depth": 0,
    "improvement_strategies": [],
    "strategy_evolution_history": []
}
```

### Consciousness Flow Metrics

```python
consciousness_flow_metrics = {
    "awareness_levels": [],
    "cognitive_resonance_score": 0.0,
    "pattern_recognition_depth": 0,
    "meta_cognitive_state_transitions": []
}
```
