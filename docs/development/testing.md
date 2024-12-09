# Testing Guide

## Overview

NanoASI uses pytest for testing. Our test suite covers:
- Unit tests
- Integration tests
- Performance tests
- Consciousness flow validation

## Running Tests

### Basic Test Run
```bash
pytest
```

### With Coverage
```bash
pytest --cov=nano_asi
```

### Performance Tests
```bash
pytest tests/performance/
```

## Writing Tests

### Test Structure
```python
import pytest
from nano_asi import ASI

@pytest.mark.asyncio
async def test_feature():
    # Arrange
    asi = ASI()
    
    # Act
    result = await asi.run("test task")
    
    # Assert
    assert result.solution is not None
```

### Test Categories

1. **Unit Tests**
   - Test individual components
   - Mock dependencies
   - Focus on edge cases

2. **Integration Tests**
   - Test component interactions
   - Validate consciousness flow
   - Check universe exploration

3. **Performance Tests**
   - Measure token efficiency
   - Track temporal ROI
   - Monitor resource usage

## Best Practices

1. **Test Organization**
   - Group related tests
   - Use descriptive names
   - Follow AAA pattern (Arrange-Act-Assert)

2. **Mocking**
   - Mock external dependencies
   - Use pytest fixtures
   - Simulate edge cases

3. **Assertions**
   - Be specific in assertions
   - Check both positive and negative cases
   - Validate data structures

## Fixtures

```python
@pytest.fixture
def asi():
    return ASI()

@pytest.fixture
def consciousness_tracker():
    return ConsciousnessTracker()
```

## Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_consciousness.py

# Run tests matching a pattern
pytest -k "consciousness"

# Run marked tests
pytest -m "slow"
```

## CI/CD Integration

Our GitHub Actions workflow runs:
1. Full test suite
2. Coverage report
3. Performance benchmarks
4. Documentation tests

## Troubleshooting

Common issues and solutions:
1. **Async Test Failures**
   - Use pytest-asyncio
   - Mark tests with @pytest.mark.asyncio

2. **Resource Cleanup**
   - Use fixtures for setup/teardown
   - Clean up temporary files

3. **Random Failures**
   - Set random seeds
   - Use deterministic test data
