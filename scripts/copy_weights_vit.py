import torch
from faeyon import X, Op
from faeyon.models import ViT
from transformers import ViTForImageClassification, AutoImageProcessor
from datasets import load_dataset


repo = "google/vit-base-patch16-224"
hf_model = ViTForImageClassification.from_pretrained(repo)
image_processor = AutoImageProcessor.from_pretrained(repo)


imagenet = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)

model = ViT(
    embed_size=768,
    heads=12,
    image_size=(224, 224),
    patch_size=16,
    num_layers=12,
    mlp_size=3072
)

model.eval()

inputs = image_processor(
    images=imagenet["train"][0]["image"],
    return_tensors="np"
)
hf_model.eval()


hf_vit = hf_model.vit

with torch.no_grad():
    hf_patch_embedding = hf_vit.embeddings.patch_embeddings.projection
    model.patch_embedding.weight.copy_(hf_patch_embedding.weight)
    model.patch_embedding.bias.copy_(hf_patch_embedding.bias)
    model.cls_token.copy_(hf_vit.embeddings.cls_token)
    model.pos_embeddings.embeddings.copy_(hf_vit.embeddings.position_embeddings[:, 1:, :].reshape([1, 14, 14, -1]).permute(0, 3, 1, 2))
    model.pos_embeddings.non_positional.copy_(hf_vit.embeddings.position_embeddings[:, 0, :].mT)

    model.lnorm.weight.copy_(hf_vit.layernorm.weight)
    model.lnorm.bias.copy_(hf_vit.layernorm.bias)
    model.classifier.weight.copy_(hf_model.classifier.weight)
    model.classifier.bias.copy_(hf_model.classifier.bias)
    blocks = model.blocks
    for i in range(12):
        hf_layer = hf_vit.encoder.layer[i]
        hf_attn = hf_layer.attention.attention
        hf_out = hf_layer.attention.output.dense
        blocks.get_parameter(f"attention.mlist.{i}.in_proj_weight").copy_(torch.cat([
            hf_attn.query.weight, 
            hf_attn.key.weight, 
            hf_attn.value.weight
        ]))
        blocks.get_parameter(f"attention.mlist.{i}.in_proj_bias").copy_(torch.cat([
            hf_attn.query.bias, 
            hf_attn.key.bias, 
            hf_attn.value.bias
        ]))
        blocks.get_parameter(f"attention.mlist.{i}.out_proj.weight").copy_(hf_out.weight)
        blocks.get_parameter(f"attention.mlist.{i}.out_proj.bias").copy_(hf_out.bias)

        blocks.get_parameter(f"linear1.mlist.{i}.weight").copy_(hf_layer.intermediate.dense.weight)
        blocks.get_parameter(f"linear1.mlist.{i}.bias").copy_(hf_layer.intermediate.dense.bias)
        blocks.get_parameter(f"linear2.mlist.{i}.weight").copy_(hf_layer.output.dense.weight)
        blocks.get_parameter(f"linear2.mlist.{i}.bias").copy_(hf_layer.output.dense.bias)
        blocks.get_parameter(f"lnorm_in.mlist.{i}.weight").copy_(hf_layer.layernorm_before.weight)
        blocks.get_parameter(f"lnorm_in.mlist.{i}.bias").copy_(hf_layer.layernorm_before.bias)
        blocks.get_parameter(f"lnorm_out.mlist.{i}.weight").copy_(hf_layer.layernorm_after.weight)
        blocks.get_parameter(f"lnorm_out.mlist.{i}.bias").copy_(hf_layer.layernorm_after.bias)

img = torch.tensor(inputs["pixel_values"])
y_hf  = hf_model(img, output_hidden_states=True)
y  = model(img)

print("Total error:")
print(abs(y -  y_hf.logits).sum())

print("\n Saved states")
print(model.fstate.collect())


torch.save(model.state_dict(), "vit.pt")


