"""Smoke tests for cookbook model builders."""

import torch
import pytest

from faeyon import Input
from faeyon.models.implementations.resnet import build_resnet50
from faeyon.models.implementations.vae import build_vae
from faeyon.models.implementations.unet import build_unet
from faeyon.models.implementations.yolo import build_yolo
from faeyon.models.implementations.qwen import build_qwen
from faeyon.models.implementations.mixtral import build_mixtral
from faeyon.models.implementations.multitask import build_multitask
from faeyon.models.implementations.deepseek import build_deepseek


class TestResNet50:
    def test_forward_shape(self):
        model = build_resnet50(num_classes=10)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)


class TestVAE:
    def test_forward_keys(self):
        model = build_vae(input_dim=64, hidden=32, latent=8)
        out = model(torch.randn(2, 64))
        assert set(out) == {"recon", "mu", "logvar"}
        assert out["recon"].shape == (2, 64)


class TestUNet:
    def test_forward_shape(self):
        model = build_unet(base=8)
        out = model(torch.randn(2, 1, 32, 32))
        assert out.shape == (2, 1, 32, 32)


class TestYOLO:
    def test_forward_scales(self):
        model = build_yolo(num_classes=3, num_anchors=2)
        out = model(torch.randn(2, 3, 64, 64))
        assert set(out) == {"small", "medium", "large"}


class TestQwen:
    def test_forward_shape(self):
        model = build_qwen(
            vocab_size=50,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
        )
        out = model(Input(ids=torch.randint(0, 50, (2, 5)), mask=None))
        assert out.shape == (2, 5, 50)


class TestMixtral:
    def test_forward_shape(self):
        model = build_mixtral(
            vocab_size=50,
            hidden_size=32,
            num_heads=4,
            num_layers=1,
            intermediate_size=32,
            n_experts=4,
            top_k=2,
        )
        out = model(Input(ids=torch.randint(0, 50, (2, 5))))
        assert out.shape == (2, 5, 50)


class TestMultiTask:
    def test_forward_heads(self):
        model = build_multitask(embed=16, vocab_size=40, n_meta=3, n_categories=5)
        out = model(
            Input(
                image=torch.randn(2, 3, 16, 16),
                text=torch.randint(0, 40, (2, 4)),
                meta=torch.randn(2, 3),
            )
        )
        assert out["category"].shape == (2, 5)
        assert out["price"].shape == (2, 1)
        assert out["quality"].shape == (2, 1)


class TestDeepSeek:
    def test_forward_shape(self):
        model = build_deepseek(
            vocab_size=50,
            hidden_size=32,
            num_heads=4,
            num_layers=1,
            intermediate_size=32,
            n_experts=4,
            top_k=2,
            kv_lora_rank=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
        )
        out = model(Input(ids=torch.randint(0, 50, (2, 5))))
        assert out.shape == (2, 5, 50)
