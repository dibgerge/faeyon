# Metrics Overview

Faeyon provides a flexible metric system for tracking training progress.

## Basic Usage

```python
from faeyon.metrics import Accuracy, MetricCollection

# Single metric
accuracy = Accuracy()
accuracy.update(predictions, targets)
result = accuracy.compute()

# Metric collection
metrics = MetricCollection([Accuracy(), MeanMetric()])
metrics.update(predictions, targets)
results = metrics.compute()
```

## Available Metrics

- **Accuracy**: Classification accuracy
- **MeanMetric**: Mean of values
- **MetricCollection**: Collection of multiple metrics

## Custom Metrics

```python
from faeyon.metrics import Metric

class MyMetric(Metric):
    def update(self, predictions, targets):
        # Update metric state
        pass
    
    def compute(self):
        # Compute metric value
        return value
```

## Learn More

See the [API Reference](../../api/metrics.md) for complete documentation.

