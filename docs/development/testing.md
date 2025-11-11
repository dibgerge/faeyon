# Testing

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=faeyon

# Run specific test file
pytest tests/test_io.py
```

## Test Structure

Tests are organized in the `tests/` directory mirroring the source structure:

```
tests/
  ├── magic/
  ├── models/
  ├── training/
  ├── metrics/
  └── nn/
```

## Writing Tests

```python
import pytest
from faeyon import X

def test_x_operations():
    x = X()
    # Your test here
    assert True
```

