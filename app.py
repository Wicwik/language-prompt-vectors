import argparse
import os
import torch
import wandb


from args import DataTrainingArguments, ArgumentParser, PromptVectorConfig

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model
from trl import SFTConfig, ModelConfig

from fastapi import FastAPI


class LLMWrapper:
    def __init__(self, config, device="cuda"):
        self.timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        self.device = device

        self._parse_args(config=config)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        self.model.active_adapters = [
            "default"
        ]  # fix because llama has some active adapters for some reason
        self.model = get_peft_model(self.model, peft_config=self.peft_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.data_args.data_tokenizer_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )

        self.tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.pipeine = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=False,
            top_p=None,
            temperature=None,
            device=self.device,
        )

    def generate(self, input_str):
        preprocessed_input = self._preprocess_input(input_str)

        result = self.pipeine(preprocessed_input)

        answer = (
            result[0]["generated_text"]
            .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
            .strip()
        )

        return answer

    def _preprocess_input(self, input_str):
        return self.tokenizer.apply_chat_template(
            [{"content": input_str, "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_args(self, config):
        parser = ArgumentParser(
            (SFTConfig, ModelConfig, DataTrainingArguments, PromptVectorConfig)
        )

        self.training_args, self.model_args, self.data_args, self.peft_config = (
            parser.parse_toml_file(config)
        )


llm = LLMWrapper("configs/slovak_alpaca.toml")
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Robo's LLM FastAPI!"}


@app.get("/generate/{input_str}")
async def generate(input_str: str):
    return {"message": llm.generate(input_str)}
