# Delayable

`Delayable` is an abstract base class for objects that can be evaluated lazily with data.

## Overview

Delayable objects can be resolved with data using the `using()` method or the `>>` operator:

```python
from faeyon.magic.spells import Delayable

class MyDelayable(Delayable):
    def _resolve(self, data):
        # Process data and return result
        return processed_data

# Use it
delayable = MyDelayable()
result = data >> delayable
# or
result = delayable.using(data)
```

## Conditional Evaluation

Delayable objects support conditional evaluation:

```python
delayable = MyDelayable()
conditional = delayable.if_(condition, else_=default_value)
```

## Chaining

Delayable objects can be chained:

```python
result = data >> delayable1 >> delayable2 >> delayable3
```

## Learn More

See the [API Reference](../../api/magic.md) for complete API documentation.

