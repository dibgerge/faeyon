# Testing

## Running tests

```bash
pytest
pytest --cov=faeyon
pytest tests/magic/test_spells.py
pytest tests/magic/test_spells.py::TestChain::test_resolve_propagates_x
```

## Layout

One test module per package module, mirroring source:

```
tests/
  magic/
    test_spells.py      # faeyon.magic.spells
    test_faek.py        # faeyon.magic.faek
    test_materialize.py # faeyon.magic.materialize
  test_modifiers.py     # faeyon.modifiers
  models/
  nn/
```

## Classes and case names

- One test class per production class: `TestChain`, `TestF`, `TestX`, …
- Case names: `test_<method>_<descriptor>` — short, method-first:

```python
class TestChain:
    def test_resolve_propagates_x(self): ...
    def test_rshift_int_suffixes_names(self): ...
    def test_rshift_named_rhs_opaque(self): ...
```

Descriptors name the behavior under test, not the scenario essay. Prefer one assert-focus per test.

## Style

- Build expressions with the DSL (`X`, `>>`, `%`, `|`); enable `faek` only when constructing from `nn.Module`.
- Prefer `materialize` / resolve over ad-hoc interpretation in magic unit tests.
- Delete obsolete commented pre-refactor tests once replaced; do not leave large commented suites.
