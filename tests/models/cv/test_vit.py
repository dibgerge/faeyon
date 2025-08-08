import torch
from faeyon.models import ViT
from faeyon.models.cv.vit import ViTBlock


class TestViTBlock:
    def test_vit_block_forward(self):
        block = ViTBlock(
            num_heads=2,
            embed_dim=4,
            mlp_size=8,
            dropout=0.0,
            lnorm_eps=1e-6,
        )
        x = torch.randn(2, 5, 4)
        out = block(x)
        assert out.shape == (2, 5, 4)


class TestViT:
    def test_vit_forward(self):
        model = ViT(
            embed_size=4,
            heads=2,
            image_size=8,
            patch_size=4,
            num_layers=2,
            mlp_size=4,
        )

        x = torch.randn(2, 3, 8, 8)
        out = model(x)
        assert out.shape == (2, 1000)
