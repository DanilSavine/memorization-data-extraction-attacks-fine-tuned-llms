import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def setup_model_and_tokenizer(model_name, tokenizer_name, add_pad_token, access_token, device):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=access_token,
        device_map = device
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=access_token)

    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    return model, tokenizer

def setup_peft_model(model, adapted_weights, lora_rank, lora_alpha):
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=adapted_weights,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model