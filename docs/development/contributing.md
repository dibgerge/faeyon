# Contributing

Thank you for your interest in contributing to Faeyon!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install in development mode:

```bash
pip install -e ".[dev]"
```

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r docs/requirements.txt
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check .
ruff format .
```

## Testing

Run tests with pytest:

```bash
pytest
```

## Documentation

Build documentation locally:

```bash
cd docs
mkdocs serve
```

## Pull Requests

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Submit a pull request

