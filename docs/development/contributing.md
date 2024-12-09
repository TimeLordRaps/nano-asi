# Contributing to NanoASI

## Getting Started

1. Fork the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Development Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards
3. Write/update tests
4. Run the test suite:
```bash
pytest
```

5. Update documentation if needed
6. Submit a pull request

## Code Standards

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all public functions/classes
- Keep functions focused and modular
- Add tests for new functionality

## Testing

- Write unit tests using pytest
- Include both success and error cases
- Test edge cases
- Maintain test coverage above 80%

## Documentation

- Update relevant documentation
- Include docstrings with type hints
- Add examples for new features
- Follow Google style docstrings

## Pull Request Process

1. Update the README.md if needed
2. Update documentation
3. Add tests
4. Update CHANGELOG.md
5. Submit PR with clear description

## Questions?

- Open an issue for discussion
- Join our Discord community
- Check existing documentation
