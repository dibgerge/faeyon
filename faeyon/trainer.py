import optax

from typing import Optional
from flax import nnx


def classification_loss(model, batch, labels):
    logits = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    ).mean()
    return loss, logits


def train(
    model,
    train_data,
    eval_data=None,
    epochs=1,
    optimizer: str = "adam",
    optimizer_kwargs: Optional[dict] = None
):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    opt = nnx.Optimizer(model, getattr(optax, optimizer)(**optimizer_kwargs))
    grad_fn = nnx.value_and_grad(classification_loss, has_aux=True)

    train_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    val_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    for epoch in range(epochs):
        for batch, labels in train_data:
            (loss, logits), grads = grad_fn(model, batch, labels)

            opt.update(grads)
            train_metrics.update(loss=loss, logits=logits, labels=labels)

        msg = f"Epoch: {epoch}"
        metric_values = train_metrics.compute()

        for metric_name, metric_value in metric_values.items():
            msg += f", train_{metric_name}: {metric_value.item():.3f}"

        if eval_data is not None:
            for batch, labels in eval_data:
                loss, logits = classification_loss(model, batch, labels)
                val_metrics.update(loss=loss, logits=logits, labels=labels)

            metric_values = val_metrics.compute()

            for metric_name, metric_value in metric_values.items():
                msg += f", val_{metric_name}: {metric_value.item():.3f}"

        print(msg)
        train_metrics.reset()
        val_metrics.reset()
