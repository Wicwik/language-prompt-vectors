import argparse
import os
import torch
import wandb


from args import DataTrainingArguments, ArgumentParser, PromptVectorConfig
from tasks import AutoTask

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig, ModelConfig


def apply_test_template(examples):
    return {
        "text": tokenizer.apply_chat_template(
            [examples], tokenize=False, add_generation_prompt=True
        )
    }


def apply_template(examples):
    return {
        "text": tokenizer.apply_chat_template(
            [examples, {"role": "assistant", "content": examples["target"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


timestamp = datetime.now().strftime("%m%d%Y%H%M%S")

argparse_parser = argparse.ArgumentParser(
    prog="Run prompt tuning",
    description="Run prompt tuning to train soft-prompts.",
)

argparse_parser.add_argument("filename", help="Filename of a config to run.")
argparse_parser.add_argument(
    "--print_data", action="store_true", help="Print parsed data and exit."
)
args = argparse_parser.parse_args()

parser = ArgumentParser(
    (SFTConfig, ModelConfig, DataTrainingArguments, PromptVectorConfig)
)

training_args, model_args, data_args, peft_config = parser.parse_toml_file(
    args.filename
)

os.environ["WANDB_PROJECT"] = "arithmetics"

output_dir = training_args.output_dir

for init_prompt in peft_config.init_prompts:
    for dataset_name in data_args.dataset_names:

        training_args.output_dir = f"{output_dir}_{timestamp}_{'_'.join(data_args.dataset_names)}_{init_prompt}"
        training_args.run_name = f"prompt_tuning_{timestamp}_{'_'.join(data_args.dataset_names)}_{init_prompt}"

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        model.active_adapters = [
            "default"
        ]  # fix because llama has some active adapters for some reason
        model = get_peft_model(model, peft_config=peft_config)

        tokenizer = AutoTokenizer.from_pretrained(
            data_args.data_tokenizer_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        model.prompt_encoder.default.embedding.weight = torch.nn.Parameter(
            torch.load(f"soft_prompts/{init_prompt}/{init_prompt}.bin")[
                "prompt_embeddings"
            ].to("cuda")
        )

        print("current PT weights:", model.prompt_encoder.default.embedding.weight)

        model.print_trainable_parameters()

        print(f"task: {dataset_name}")

        train_dataset = AutoTask.get(dataset_name).get(
            split="train",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_train_samples,
            split_validation_test=data_args.split_validation_test,
        )
        valid_dataset = AutoTask.get(dataset_name).get(
            split="validation",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_valid_samples,
            split_validation_test=data_args.split_validation_test,
        )
        test_dataset = AutoTask.get(dataset_name).get(
            split="test",
            task_type=peft_config.task_type,
            add_prefix=False,
            n_obs=data_args.max_test_samples,
            split_validation_test=data_args.split_validation_test,
        )

        chat_train_dataset = train_dataset.map(apply_template)
        chat_valid_dataset = valid_dataset.map(apply_template)
        chat_test_dataset = test_dataset.map(apply_test_template)

        if args.print_data:
            print("Train data")
            print(chat_train_dataset["text"][0])

            print("Valid data")
            print(chat_valid_dataset["text"][0])

            print("Test data")
            print(chat_test_dataset["text"][0])

            exit(0)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=chat_train_dataset,
            eval_dataset=chat_valid_dataset,
            tokenizer=tokenizer,
            packing=False,
        )

        trainer.train()

        if isinstance(dataset_name, list):
            save_name = f"./saves/prompt_tuning_{timestamp}_{'_'.join(dataset_name)}_{init_prompt}_best"
        else:
            save_name = (
                f"./saves/prompt_tuning_{timestamp}_{dataset_name}_{init_prompt}_best"
            )

        model.save_pretrained(save_name)

        if wandb.run is not None:
            artifact = wandb.Artifact(name=training_args.run_name, type="weights")
            artifact.add_dir(local_path=save_name)
            wandb.run.log_artifact(artifact)
            wandb.log(data={})

            wandb.finish()
