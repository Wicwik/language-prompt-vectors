import torch


from args import DataTrainingArguments, ArgumentParser, PromptVectorConfig

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from peft import get_peft_model, PeftModel
from trl import SFTConfig, ModelConfig

from typing import List

from fastapi import FastAPI

from utils import load_prompt, load_prompt_vector

from pydantic import BaseModel


class Params(BaseModel):
    input_str: str
    soft_prompts: List[str] = []


SOFT_PROMPT_PATHS = {
    "SlovakAlpaca": {
        "soft_prompt": "soft_prompts/origin_0_meta-llama-3.1-8b-instruct/slovak_alpaca.safetensors",
        "init_prompt": "soft_prompts/origin_0_meta-llama-3.1-8b-instruct/origin_0_meta-llama-3.1-8b-instruct.bin",
    },
    "SST2": {
        "soft_prompt": "soft_prompts/origin_0_meta-llama-3.1-8b-instruct/sst2.bin",
        "init_prompt": "soft_prompts/origin_0_meta-llama-3.1-8b-instruct/origin_0_meta-llama-3.1-8b-instruct.bin",
    },
}


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

        self.pipeline = self._create_pipeline(self.model)

    def _create_pipeline(self, model) -> Pipeline:
        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            temperature=None,
            use_cache=False,
            device=self.device,
        )

    def _set_soft_prompt(self, soft_prompts: List[str]):
        print(soft_prompts)

        if soft_prompts == [] and isinstance(self.pipeline.model, PeftModel):
            print("is PEFT MODEL")
            self.pipeline = self._create_pipeline(self.model.base_model)

        elif soft_prompts != []:
            prompt_vectors = []

            for soft_prompt_name in soft_prompts:
                prompt_vectors.append(
                    load_prompt_vector(
                        soft_prompt_name,
                        SOFT_PROMPT_PATHS[soft_prompt_name]["soft_prompt"],
                        SOFT_PROMPT_PATHS[soft_prompt_name]["init_prompt"],
                    )
                )

            self.model.prompt_encoder.default.embedding.weight = sum(
                prompt_vectors
            ).apply(
                load_prompt(SOFT_PROMPT_PATHS[soft_prompts[0]]["init_prompt"]).to(
                    self.device
                )
            )

            self.pipeline = self._create_pipeline(self.model)

    def generate(self, input_str, soft_prompts):
        preprocessed_input = self._preprocess_input(input_str)

        self._set_soft_prompt(soft_prompts)
        result = self.pipeline(preprocessed_input)

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


@app.post("/generate")
async def generate(params: Params):
    return {"message": llm.generate(params.input_str, params.soft_prompts)}
