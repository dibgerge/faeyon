from flax import nnx

from faeyon.cv import ViT
from transformers import FlaxViTForImageClassification, AutoImageProcessor
from datasets import load_dataset

repo = "google/vit-base-patch16-224"
hf_model = FlaxViTForImageClassification.from_pretrained(repo)
image_processor = AutoImageProcessor.from_pretrained(repo)


imagenet = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)

model = ViT(
    heads=12,
    image_height=224,
    image_width=224,
    patch_size=16,
    layers=12,
    hidden_size=768,
    mlp_size=3072,
    rngs=nnx.Rngs(0)
)
model.eval()

inputs = image_processor(
    images=imagenet["train"][0]["image"],
    return_tensors="np"
)


def copy_values(copy_to, copy_from, keys=None, shape=None):

    if keys is None:
        copy_to = {"data": copy_to}
        copy_from = {"data": copy_from}
        keys = ["data"]

    for key in keys:

        if shape is not None:
            src = copy_from[key].reshape(shape)
        else:
            src = copy_from[key]
        
        if isinstance(copy_to, dict):
            dest = copy_to[key]
        else:
            dest = getattr(copy_to, key)
        
        if dest.shape != src.shape:
            raise ValueError(
                f"{dest.__class__.__name__} and {src.__class__.__name__} must"
                f" have the same shape. {dest.shape} != {src.shape}."
            )
        
        dest.value = dest.value.at[:].set(src)


def copy_dense(copy_to, copy_from, in_shape=None, out_shape=None):
    kernel = copy_from["kernel"]
    bias = copy_from["bias"]

    if in_shape is not None and out_shape is not None:

        if isinstance(in_shape, int):
            in_shape = (in_shape,)

        if isinstance(out_shape, int):
            out_shape = (out_shape,)

        kernel_shape = in_shape + out_shape
        bias_shape = out_shape
    else:
        kernel_shape = None
        bias_shape = None

    copy_values(copy_to.kernel, kernel, shape=kernel_shape)
    copy_values(copy_to.bias, bias, shape=bias_shape)


hf_params = hf_model.params["vit"]
hf_patch_embedding = hf_params["embeddings"]["patch_embeddings"]["projection"]

patch_embedding = model.patch_embedding

copy_values(model.cls_token, hf_params["embeddings"]["cls_token"])
copy_dense(patch_embedding, hf_patch_embedding)

copy_values(model.positional_embedding, hf_params["embeddings"]["position_embeddings"].squeeze())


for i, hf_layer in hf_params["encoder"]["layer"].items():
    i = int(i)
    layer = model.layers[i]
    attention = layer.attention
    hf_attention = hf_layer["attention"]["attention"]

    copy_dense(attention.key, hf_attention["key"], in_shape=768, out_shape=(12, 64))
    copy_dense(attention.query, hf_attention["query"], in_shape=768, out_shape=(12, 64))
    copy_dense(attention.value, hf_attention["value"], in_shape=768, out_shape=(12, 64))
    copy_dense(attention.out, hf_layer["attention"]["output"]["dense"], in_shape=(12, 64), out_shape=768)

    copy_values(layer.lnorm_in, hf_layer["layernorm_before"], keys=["scale", "bias"])
    copy_values(layer.lnorm_out, hf_layer["layernorm_after"], keys=["scale", "bias"])

    copy_dense(layer.linear1, hf_layer["intermediate"]["dense"])
    copy_dense(layer.linear2, hf_layer["output"]["dense"])

copy_values(model.lnorm, hf_params["layernorm"], keys=["scale", "bias"])
copy_dense(model.classifier, hf_model.params["classifier"])


x = inputs["pixel_values"].transpose((0, 2, 3, 1))
y = model(x)
hf_y = hf_model(**inputs)


print(abs(hf_y.logits.squeeze() - y.squeeze()).mean())
