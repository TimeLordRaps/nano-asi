"""Tests for ConsciousnessTracker module."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from nano-asi.modules import ConsciousnessTracker

@pytest.mark.asyncio
async def test_consciousness_tracker_initialization(consciousness_tracker):
    """Test ConsciousnessTracker initialization."""
    assert len(consciousness_tracker.states) == 0
    assert isinstance(consciousness_tracker.pattern_evolution, dict)
    assert isinstance(consciousness_tracker.meta_cognitive_state, dict)

@pytest.mark.asyncio
async def test_track_consciousness(consciousness_tracker, sample_state_data):
    """Test consciousness state tracking."""
    state = await consciousness_tracker.track_consciousness(sample_state_data)
    
    # Verify state structure
    assert state.timestamp > 0
    assert len(state.activation_patterns) > 0
    assert len(state.thought_chains) > 0
    assert isinstance(state.meta_insights, list)
    
    # Verify state was added to history
    assert len(consciousness_tracker.states) == 1
    assert consciousness_tracker.states[0] == state

@pytest.mark.asyncio
async def test_analyze_activations(consciousness_tracker, sample_state_data):
    """Test neural activation pattern analysis."""
    patterns = await consciousness_tracker._analyze_activations(sample_state_data)
    
    assert len(patterns) == len(sample_state_data['activations'])
    for pattern in patterns:
        assert 'layer_type' in pattern
        assert 'activation_stats' in pattern
        assert 'pattern_type' in pattern
        assert 'quantum_stats' in pattern
        assert 'coherence' in pattern
        assert 'entanglement' in pattern
        assert 'resonance_score' in pattern

@pytest.mark.asyncio
async def test_extract_thought_chains(consciousness_tracker, sample_state_data):
    """Test thought chain extraction and analysis."""
    chains = await consciousness_tracker._extract_thought_chains(sample_state_data)
    
    assert len(chains) == len(sample_state_data['thoughts'])
    for chain in chains:
        assert 'content' in chain
        assert 'dependencies' in chain
        assert 'meta_level' in chain

@pytest.mark.asyncio
async def test_analyze_meta_patterns(consciousness_tracker, sample_state_data):
    """Test meta-pattern analysis."""
    # First add some state history
    await consciousness_tracker.track_consciousness(sample_state_data)
    
    insights = await consciousness_tracker._analyze_meta_patterns(sample_state_data)
    assert isinstance(insights, list)
    
    if insights:  # Will have insights after state history exists
        insight = insights[0]
        assert 'pattern_metrics' in insight
        assert 'effectiveness' in insight
        assert 'improvement_suggestions' in insight

@pytest.mark.asyncio
async def test_empty_state_handling(consciousness_tracker):
    """Test handling of empty state data."""
    empty_state = {}
    state = await consciousness_tracker.track_consciousness(empty_state)
    
    # Should create valid state even with empty input
    assert state.timestamp > 0
    assert isinstance(state.activation_patterns, list)
    assert isinstance(state.thought_chains, list)
    assert isinstance(state.meta_insights, list)

@pytest.mark.asyncio
async def test_pattern_evolution_tracking(consciousness_tracker, sample_state_data):
    """Test tracking of pattern evolution over multiple states."""
    # Track multiple states
    states = []
    for _ in range(3):
        state = await consciousness_tracker.track_consciousness(sample_state_data)
        states.append(state)
    
    # Verify evolution tracking
    assert len(consciousness_tracker.states) == 3
    assert all(s in consciousness_tracker.states for s in states)
    
    # Analyze evolution
    metrics = consciousness_tracker._analyze_pattern_evolution()
    assert isinstance(metrics, dict)
    assert 'improvement_rate' in metrics
    assert 'pattern_stability' in metrics
    assert 'consciousness_coherence' in metrics
