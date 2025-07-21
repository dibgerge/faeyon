import abc
import enum
from torch import nn
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


class FaeModule(LightningModule, abc.ABC):
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

    def save(self, path: str):
        """ 
        Saves the current model and its weights to the given directory, which can be local
        or remote directory (e.g. hugging face, s3, etc)
        """
        pass


class Classification(FaeModule):
    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model

        # self.train_metrics = nnx.MultiMetric(
        #     accuracy=nnx.metrics.Accuracy(),
        #     loss=nnx.metrics.Average('loss'),
        # )
        # self.val_metrics = nnx.MultiMetric(
        #     accuracy=nnx.metrics.Accuracy(),
        #     loss=nnx.metrics.Average('loss'),
        # )

    def __call__(self, x):
        return self.model(x)

    def train_step(self, batch):
        # print(batch)
        logits = self(batch["data"])
        # loss = optax.softmax_cross_entropy_with_integer_labels(
        #     logits=logits,
        #     labels=batch["label"]
        # ).mean()
        # return loss

    def val_step(self, batch):
        # logits = self.model(batch["data"])
        # loss = optax.softmax_cross_entropy_with_integer_labels(
        #     logits=logits,
        #     labels=batch["label"]
        # ).mean()

        # self.val_metrics.update(loss=loss, logits=logits, labels=batch["label"])
        pass

    def on_epoch_end(self):
        pass
        # print(self.train_metrics.compute())
        # print(self.val_metrics.compute())
        # self.train_metrics.reset()
        # self.val_metrics.reset()

    # def optimizers(self):
        # return nnx.Optimizer(self.model, optax.adam(learning_rate=0.001))


# if __name__ == "__main__":
#     from cv import ViT
#     model = ViT(
#         heads=12,
#         image_height=224,
#         image_width=224,
#         patch_size=16,
#         layers=12,
#         hidden_size=768,
#         mlp_size=3072,
#     )

#     spell = Classification(
#         model=model,
#     )

#     spell.save("/home/dibgerge/Documents/projects/faeyon-ml/experiments/")
