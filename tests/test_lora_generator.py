"""Tests for LoRA generator module with temporal tracking."""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from nano-asi.modules import LoRAGenerator

@pytest.mark.asyncio
async def test_lora_generator_initialization(lora_generator):
    """Test LoRAGenerator initialization."""
    assert lora_generator.hyperparameters['lora_r'] > 0
    assert lora_generator.hyperparameters['lora_alpha'] > 0
    assert 0 <= lora_generator.hyperparameters['lora_dropout'] <= 1
    assert isinstance(lora_generator.pattern_evolution_history, list)
    assert isinstance(lora_generator.consciousness_flow, list)

@pytest.mark.asyncio
async def test_generate_lora_adapter(lora_generator, sample_conditional_tokens):
    """Test basic LoRA adapter generation."""
    adapter = await lora_generator.generate_lora_adapter(sample_conditional_tokens)
    
    # Verify adapter structure
    assert isinstance(adapter, dict)
    assert 'params' in adapter
    assert 'consciousness_flow' in adapter
    assert 'universe_results' in adapter
    
    # Verify parameter shapes
    params = adapter['params']
    assert all(isinstance(v, torch.Tensor) for v in params.values())
    assert all(v.shape[0] == lora_generator.hyperparameters['lora_r'] for v in params.values())

@pytest.mark.asyncio
async def test_explore_parallel_universes(lora_generator, sample_conditional_tokens):
    """Test parallel universe exploration for LoRA generation."""
    num_universes = 3
    results = await lora_generator.explore_parallel_universes(num_universes)
    
    assert len(results['results']) == num_universes
    assert 'patterns' in results
    assert 'consciousness_states' in results
    assert 'best_universe' in results
    
    # Verify universe results structure
    for result in results['results']:
        assert 'universe_id' in result
        assert 'params' in result
        assert 'consciousness_flow' in result
        assert 'quantum_resonance' in result

@pytest.mark.asyncio
async def test_optimize_consciousness_flow(lora_generator, sample_flow_data):
    """Test consciousness flow optimization."""
    flow_data = await lora_generator.optimize_consciousness_flow(sample_flow_data)
    
    # Verify optimization results
    assert isinstance(flow_data, dict)
    assert 'patterns' in flow_data
    assert 'consciousness_states' in flow_data
    assert 'activation_patterns' in flow_data
    assert 'quantum_resonance' in flow_data

@pytest.mark.asyncio
async def test_meta_optimize(lora_generator, validation_data):
    """Test meta-optimization capabilities."""
    results = await lora_generator.meta_optimize(validation_data)
    
    assert 'final_performance' in results
    assert 'optimization_history' in results
    assert isinstance(results['optimization_history'], list)
    assert len(results['optimization_history']) > 0
    
    # Verify optimization metrics
    history = results['optimization_history']
    for entry in history:
        assert 'iteration' in entry
        assert 'best_score' in entry
        assert 'candidates' in entry

@pytest.mark.asyncio
async def test_recursive_improvement(lora_generator, sample_conditional_tokens):
    """Test recursive self-improvement capabilities with temporal tracking."""
    # Generate initial adapter
    initial_adapter = await lora_generator.generate_lora_adapter(sample_conditional_tokens)
    
    # Track token evolution over multiple time steps
    token_states: List[torch.Tensor] = []
    temporal_coherence_scores: List[float] = []
    
    # Improve adapter recursively across time steps
    improved_adapter = await lora_generator.recursive_improve(initial_adapter)
    for step in range(len(improved_adapter['improvement_history'])):
        # Track token state at this time step
        token_states.append(improved_adapter['improvement_history'][step]['token_state'])
        
        # Measure temporal coherence between consecutive states
        if step > 0:
            coherence = torch.nn.functional.cosine_similarity(
                token_states[step].flatten(),
                token_states[step-1].flatten(),
                dim=0
            )
            temporal_coherence_scores.append(float(coherence))
    
    # Verify improvements and temporal properties
    assert improved_adapter['params'] != initial_adapter['params']
    assert 'improvement_history' in improved_adapter
    assert len(improved_adapter['improvement_history']) > 0
    
    # Verify temporal coherence
    assert len(temporal_coherence_scores) > 0
    assert all(0.5 <= score <= 1.0 for score in temporal_coherence_scores), \
        "Temporal coherence should remain reasonably high between steps"
    
    # Verify monotonic improvement
    scores = [step['score'] for step in improved_adapter['improvement_history']]
    assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), \
        "Performance should improve monotonically through token-time"

@pytest.mark.asyncio
async def test_error_handling(lora_generator):
    """Test error handling and recovery."""
    # Test with invalid tokens
    with pytest.raises(ValueError):
        await lora_generator.generate_lora_adapter(None)
    
    # Test with empty tokens
    with pytest.raises(ValueError):
        await lora_generator.generate_lora_adapter(torch.tensor([]))
    
    # Test with invalid universe count
    with pytest.raises(ValueError):
        await lora_generator.explore_parallel_universes(0)

@pytest.mark.asyncio
async def test_consciousness_integration(lora_generator, consciousness_tracker, sample_conditional_tokens):
    """Test integration with consciousness tracking and temporal ordering."""
    adapter = await lora_generator.generate_lora_adapter(
        sample_conditional_tokens,
        consciousness_tracker=consciousness_tracker
    )
    
    # Verify consciousness tracking integration
    assert len(consciousness_tracker.states) > 0
    assert 'consciousness_flow' in adapter
    assert len(adapter['consciousness_flow']) > 0
    
    # Verify temporal ordering of consciousness states
    states = consciousness_tracker.states
    timestamps = [state.timestamp for state in states]
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), \
        "Consciousness states must maintain strict temporal ordering"
    
    # Verify temporal coherence of thought chains
    for i in range(1, len(states)):
        prev_chains = states[i-1].thought_chains
        curr_chains = states[i].thought_chains
        
        # Verify thought evolution follows causal dependencies
        for curr_chain in curr_chains:
            if curr_chain['dependencies']:
                assert any(
                    dep_id in [pc['id'] for pc in prev_chains]
                    for dep_id in curr_chain['dependencies']
                ), "Thought chains must respect causal temporal dependencies"
