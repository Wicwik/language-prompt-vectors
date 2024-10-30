import torch

from safetensors.torch import safe_open
from prompt_vector import PromptVector

def load_prompt(path: str) -> torch.Tensor:
    if path.lower().endswith(".safetensors"):
        prompt = load_from_safetensor(path)
    elif path.lower().endswith(".bin"):
        prompt = load_from_binary(path)
    else:
        raise ValueError(f"Unrecognized file extension: {path.split('.')[-1]}")
    
    return prompt

def load_from_safetensor(path: str) -> torch.Tensor:
    tensors = {}

    with safe_open(f"{path}/adapter_model.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return tensors["prompt_embeddings"]

def load_from_binary(path: str) -> torch.Tensor:
    return torch.load(path)["prompt_embeddings"]

def load_prompt_vector(save_path: str, init_path: str):
    task_prompt = load_prompt(save_path)
    init_prompt = load_prompt(init_path)

    prompt_vector = PromptVector(task_prompt, init_prompt)

    return prompt_vector