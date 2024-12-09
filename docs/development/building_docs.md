# Building Documentation

## Overview

NanoASI uses Sphinx for documentation with:
- Markdown support via myst-parser
- API documentation via sphinx-autodoc
- Beautiful theme via sphinx-book-theme

## Setup

1. Install documentation dependencies:
```bash
pip install -e ".[docs]"
```

2. Navigate to docs directory:
```bash
cd docs
```

## Building Documentation

### HTML Documentation
```bash
make html
```

### PDF Documentation
```bash
make latexpdf
```

### Clean Build
```bash
make clean
make html
```

## Documentation Structure

```
docs/
├── index.md              # Main landing page
├── quickstart.md         # Getting started guide
├── tutorials/            # Step-by-step guides
├── api_reference/        # API documentation
├── advanced_topics/      # Deep dives
└── development/          # Development guides
```

## Writing Documentation

### Markdown Files

Use MyST Markdown with:
- Code blocks with syntax highlighting
- Math equations via LaTeX
- Cross-references
- Admonitions

Example:
````markdown
# Component Guide

```{note}
Important implementation details.
```

## Usage
```python
from nano_asi import ASI
asi = ASI()
```
````

### API Documentation

Use Google style docstrings:
```python
def function(arg: str) -> int:
    """Short description.
    
    Longer description with details.
    
    Args:
        arg: Description of argument
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of error case
    """
```

## Local Preview

1. Build documentation:
```bash
make html
```

2. Open in browser:
```bash
python -m http.server -d _build/html
```

3. Visit http://localhost:8000

## Deployment

Documentation is automatically:
1. Built on each push
2. Deployed to GitHub Pages
3. Updated on ReadTheDocs

## Style Guide

1. **Headers**
   - Use sentence case
   - Maximum 3 levels deep
   - Include section numbers

2. **Code Examples**
   - Include imports
   - Show complete examples
   - Add expected output

3. **Cross-References**
   - Use relative links
   - Include section titles
   - Check link validity

## Tips

1. **Preview Changes**
   - Build docs locally
   - Check all pages
   - Verify cross-references

2. **Images**
   - Use SVG when possible
   - Include alt text
   - Optimize file size

3. **Performance**
   - Minimize image sizes
   - Use lazy loading
   - Cache build artifacts
