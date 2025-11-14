import torch
from faeyon.models.tasks import ClassifyTask
from faeyon.models import ViT, Pipeline
from transformers import ViTForImageClassification, AutoImageProcessor
from datasets import load_dataset
from faeyon.io import save

repo = "google/vit-base-patch16-224"

# Load the Hugging Face model
hf_model = ViTForImageClassification.from_pretrained(repo)
hf_model.eval()


# Load the Faeyon model
model = ViT(
    embed_size=768,
    heads=12,
    image_size=(224, 224),
    patch_size=16,
    num_layers=12,
    mlp_size=3072
)

pipeline = Pipeline(
    model=model,
    task=ClassifyTask(num_hidden=768, num_labels=1000, pooling=0)
)
pipeline.eval()


@torch.no_grad()
def copy_weights(hf_model, pipeline):
    model = pipeline.model
    task = pipeline.task

    hf_vit = hf_model.vit
    hf_patch_embedding = hf_vit.embeddings.patch_embeddings.projection
    model.patch_embedding.weight.copy_(hf_patch_embedding.weight)
    model.patch_embedding.bias.copy_(hf_patch_embedding.bias)
    model.cls_token.copy_(
        hf_vit.embeddings.cls_token
        + hf_vit.embeddings.position_embeddings[:, :1, :]
    )
    model.pos_embeddings.embeddings.copy_(
        hf_vit
        .embeddings
        .position_embeddings[:, 1:, :]
        .reshape([1, 14, 14, -1])
        .permute(0, 3, 1, 2)
    )
    # model.pos_embeddings.non_positional.copy_(hf_vit.embeddings.position_embeddings[:, 0, :].mT)

    model.lnorm.weight.copy_(hf_vit.layernorm.weight)
    model.lnorm.bias.copy_(hf_vit.layernorm.bias)
    task.classifier.weight.copy_(hf_model.classifier.weight)
    task.classifier.bias.copy_(hf_model.classifier.bias)
    blocks = model.blocks
    for i in range(12):
        hf_layer = hf_vit.encoder.layer[i]
        hf_attn = hf_layer.attention.attention
        hf_out = hf_layer.attention.output.dense
        blocks.get_parameter(f"attention.{i}.in_proj_weight").copy_(torch.cat([
            hf_attn.query.weight, 
            hf_attn.key.weight, 
            hf_attn.value.weight
        ]))
        blocks.get_parameter(f"attention.{i}.in_proj_bias").copy_(torch.cat([
            hf_attn.query.bias, 
            hf_attn.key.bias, 
            hf_attn.value.bias
        ]))
        blocks.get_parameter(f"attention.{i}.out_proj.weight").copy_(hf_out.weight)
        blocks.get_parameter(f"attention.{i}.out_proj.bias").copy_(hf_out.bias)

        blocks.get_parameter(f"linear1.{i}.weight").copy_(hf_layer.intermediate.dense.weight)
        blocks.get_parameter(f"linear1.{i}.bias").copy_(hf_layer.intermediate.dense.bias)
        blocks.get_parameter(f"linear2.{i}.weight").copy_(hf_layer.output.dense.weight)
        blocks.get_parameter(f"linear2.{i}.bias").copy_(hf_layer.output.dense.bias)
        blocks.get_parameter(f"lnorm_in.{i}.weight").copy_(hf_layer.layernorm_before.weight)
        blocks.get_parameter(f"lnorm_in.{i}.bias").copy_(hf_layer.layernorm_before.bias)
        blocks.get_parameter(f"lnorm_out.{i}.weight").copy_(hf_layer.layernorm_after.weight)
        blocks.get_parameter(f"lnorm_out.{i}.bias").copy_(hf_layer.layernorm_after.bias)


copy_weights(hf_model, pipeline)

# Get an example image for testing
image_processor = AutoImageProcessor.from_pretrained(repo)
imagenet = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
inputs = image_processor(
    images=imagenet["train"][0]["image"],
    return_tensors="np"
)
img = torch.tensor(inputs["pixel_values"])

y_hf  = hf_model(img, output_hidden_states=False)
y  = pipeline(img)

print("Total error:")
print(abs(y -  y_hf.logits).sum())
print("\n Saved states")
print(pipeline.fstate.collect())

print(y.argmax(), y_hf.logits.argmax())


# if torch.allclose(y, y_hf.logits, atol=1e-4, rtol=1e-8):
#     print("[SUCCESS] The saved model is the same as the original model.")
#     state_file = "hf://dibgerges/faeyon/vit/vit-base-patch16-224.pt"
#     save(
#         pipeline, 
#         "faeyon/models/configs/vit/vit-base-patch16-224.yaml", 
#         save_state=state_file
#     )
# else:
#     print("[ERROR] The saved model is not the same as the original model.")



