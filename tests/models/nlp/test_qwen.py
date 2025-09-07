import pytest
import torch
from faeyon.models import Qwen


@pytest.fixture
def model():
    return Qwen(
        vocab_size=10, 
        hidden_size=20,
        num_heads=2, 
        num_layers=2, 
        padding_idx=0,
        intermediate_size=10
    )


class TestQwen:
    def test_usage(self, model):
        model = model.cuda()
        x = torch.randint(0, 10, (1, 5), device="cuda")
        y = model(x)
        assert y.shape == (1, 5, 20)

    def test_hf_compare(self):
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        model = Qwen(
            vocab_size=151936,
            hidden_size=1024,
            num_heads=16,
            num_heads_kv=8,
            num_layers=28,
            dropout=0.0,
            intermediate_size=3072,
            padding_idx=None
        ).cuda()
        model.eval()

        model_name = "Qwen/Qwen3-0.6B"

        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # prepare the model input
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,

            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
        y = model(model_inputs["input_ids"])
