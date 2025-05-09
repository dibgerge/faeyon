import abc
import enum

import optax
from flax import nnx
from lightning import LightningModule


class Stage(enum.Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


def loss_fn(model, batch):
    # logits = model(batch['data'])

    # print("logits", logits.shape)
    # loss = optax.softmax_cross_entropy_with_integer_labels(
    #     logits=logits,
    #     labels=batch['label']
    # ).mean()
    # return loss
    # return func(batch)
    return model.train_step(batch)


class Spell(nnx.Module, abc.ABC):
    def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs

    @abc.abstractmethod
    def __call__(self, batch):
        pass
    
    @abc.abstractmethod
    def train_step(self, batch):
        pass

    @abc.abstractmethod
    def val_step(self, batch):
        pass

    @abc.abstractmethod
    def optimizers(self):
        pass

    def fit(self, train_data, eval_data=None, epochs=1):
        optimizer = self.optimizers()
        grad_fn = nnx.value_and_grad(loss_fn)

        for epoch in range(epochs):
            for batch in train_data:
                loss, grads = grad_fn(self, batch)
                print(grads)
                optimizer.update(grads)


class ClassificationSpell(Spell):
    def __init__(
        self,
        model: nnx.Module,
        rngs: nnx.Rngs
    ):
        super().__init__(rngs=rngs)
        self.model = model

        self.train_metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
        )
        self.val_metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
        )

    def __call__(self, x):
        return self.model(x)

    def train_step(self, batch):
        # print(batch)
        logits = self(batch["data"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["label"]
        ).mean()
        # print(loss)
        # self.train_metrics.update(loss=loss, logits=logits, labels=batch["label"])
        return loss

    def val_step(self, batch):
        logits = self.model(batch["data"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch["label"]
        ).mean()

        self.val_metrics.update(loss=loss, logits=logits, labels=batch["label"])

    def on_epoch_end(self):
        print(self.train_metrics.compute())
        print(self.val_metrics.compute())
        self.train_metrics.reset()
        self.val_metrics.reset()

    def optimizers(self):
        return nnx.Optimizer(self.model, optax.adam(learning_rate=0.001))
