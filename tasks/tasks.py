import functools
import datasets
import numpy as np
import torch

from .type import AutoType

from collections import OrderedDict
from typing import Mapping

from typing import List, Dict, Callable

from evaluate import Metric
from datasets import Dataset

from metrics import exact_match, macro_f1, f1

from transformers import EvalPrediction


class AbstractTask:
    name: str = NotImplemented
    preprocessor = NotImplemented
    formater = NotImplemented
    metrics: List[Metric] = NotImplemented
    metric_names: List[str] = NotImplemented
    config = NotImplemented
    dataset_config_name = NotImplemented
    seed = NotImplemented
    labels_list = None
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    small_datasets_without_all_splits = ["trec", "wnli", "sst5", "slovak_alpaca"]
    large_data_without_all_splits = [
        "qnli",
        "sst2",
        "mnli",
        "yelp_polarity",
        "dbpedia",
        "scitail",
        "snli",
        "ag_news",
        "yahoo",
        "imdb",
    ]
    id2label = NotImplemented
    label_column_name = NotImplemented

    def __init__(self, seed=42):
        self.dataset_config_name = "en"
        self.seed = seed

    def postprocessor(self, preds, labels, tokenizer):
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, return_tensors="pt"
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, return_tensors="pt"
        )

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return decoded_preds, decoded_labels

    # get maximum token lenght from labels
    def get_max_target_length(self, tokenizer, default_max_length):
        if self.labels_list is not None:
            return max([len(tokenizer.encode(label)) for label in self.labels_list])
        return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
        return n_obs

    # generates indices of the dataset randomly with seed (if same seed and data provided we will still get the same shuffle, no matter how many times initialized)
    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset: Dataset, n_obs=None):
        num_samples = len(dataset)

        if n_obs >= num_samples:
            return dataset

        return dataset.train_test_split(
            train_size=n_obs / num_samples,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
        )["train"]

    def get_splits(self, split, dataset: Dataset, validation_size):
        if split == "validation":
            return dataset.train_test_split(
                train_size=validation_size,
                test_size=1 - validation_size,
                seed=self.seed,
                stratify_by_column=self.label_column_name,
                shuffle=True,
            )["train"]

        return dataset.train_test_split(
            train_size=validation_size,
            test_size=1 - validation_size,
            seed=self.seed,
            stratify_by_column=self.label_column_name,
            shuffle=True,
        )["test"]

    def map_dataset(self, dataset: Dataset, add_prefix: bool) -> Dataset:
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc=f"Running {self.name}_preprocessor on dataset",
        )

    def load_dataset(self, split: int) -> Dataset:
        return datasets.load_dataset(
            self.name, self.dataset_config_name, split=split, script_version="master"
        )

    def get_compute_metrics(
        self, tokenizer, task_type
    ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics(eval_preds: EvalPrediction) -> Dict:
            preds, labels = eval_preds

            preds[preds == -100] = tokenizer.pad_token_id
            labels[labels == -100] = tokenizer.pad_token_id

            decoded_preds, decoded_labels = self.postprocessor(preds, labels, tokenizer)

            if task_type == "CAUSAL_LM":
                decoded_preds = [
                    dpred.split("label: ")[1] if "label: " in dpred else dpred
                    for dpred in tokenizer.batch_decode(preds, skip_special_tokens=True)
                ]

            print("compute_metrics:", decoded_preds)

            metrics = {}
            # TODO: to get rid of the zip, make classes from metrics and add metric name to it
            for m, n in zip(self.metrics, self.metric_names):
                if "f1" in n:
                    metrics.update(m(decoded_preds, decoded_labels, self.labels_list))
                else:
                    metrics.update(m(decoded_preds, decoded_labels))

            return metrics

        return compute_metrics

    def get(
        self,
        split,
        task_type="SEQ_2_SEQ_LM",
        add_prefix=True,
        n_obs=None,
        split_validation_test=False,
    ) -> Dataset:
        self.formater = AutoType.get(task_type).formater
        if (
            split_validation_test
            and self.name.replace("_text", "").replace("_instruct", "")
            in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            dataset = self.get_splits(split, dataset, 0.5)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        elif (
            split_validation_test
            and self.name.replace("_text", "").replace("_instruct", "")
            in self.large_data_without_all_splits
            and split != "test"
        ):
            dataset = self.load_dataset(split="train")
            dataset = self.get_splits(split, dataset, 1000 / len(dataset))

            if n_obs:
                dataset = self.subsample(dataset, n_obs)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split).shuffle(seed=self.seed)

            if n_obs:
                dataset = self.subsample(dataset, n_obs)

        return self.map_dataset(dataset, add_prefix)


# Sentiment classification
class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "sst2", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence: {example['sentence']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["negative", "positive"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {0: "negative", 1: "positive"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("yelp_polarity")[split]

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the sentence into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"sentence: {example['text']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]
        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


# Natural language inference
class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["entailment", "not entailment"]
    metrics = [exact_match, f1]
    metric_names = ["exact_match", "f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "not entailment"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "qnli", split=split)

    def preprocessor(self, example, add_prefix=False):
        input_texts = [
            f"Classify the question and sentence pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"question: {example['question']}",
            f"sentence: {example['sentence']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["entailment", "neutral", "contradiction"]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }
    label_column_name = "label"
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("glue", "mnli", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the premise and hypothesis pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"premise: {example['premise']}",
            f"hypothesis: {example['hypothesis']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


# Multi task question classification
class TREC(AbstractTask):
    name = "trec"
    labels_list = [
        "Abbreviation",
        "Entity",
        "Description",
        "Person",
        "Location",
        "Quantity",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "validation": "test"}
    label_column_name = "coarse_label"
    id2label = {
        0: "Abbreviation",
        1: "Entity",
        2: "Description",
        3: "Person",
        4: "Location",
        5: "Quantity",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("trec", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the question into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"question: {example['text']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class DBPEDIA(AbstractTask):
    name = "dbpedia"
    labels_list = [
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean Of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ]
    metrics = [exact_match, macro_f1]
    metric_names = ["exact_match", "macro_f1"]
    split_to_data_split = {"train": "train", "test": "test"}
    label_column_name = "label"
    id2label = {
        0: "Company",
        1: "Educational Institution",
        2: "Artist",
        3: "Athlete",
        4: "Office Holder",
        5: "Mean Of Transportation",
        6: "Building",
        7: "Natural Place",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "Written Work",
    }

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("dbpedia_14", split=split)

    def preprocessor(self, example, add_prefix=True):
        input_texts = [
            f"Classify the title and content pair into labels: {', '.join(self.labels_list)}. Reply only the corresponding label.",
            f"title: {example['title']}",
            f"content: {example['content']}",
        ]
        label_texts = [self.id2label[example[self.label_column_name]]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
        )


class SlovakAlpaca(AbstractTask):
    name = "slovak_alpaca"
    split_to_data_split = {
        "train": "train",
        "validation": "test",
        "test": "test",
    }
    label_column_name = None

    def load_dataset(self, split) -> Dataset:
        return datasets.load_dataset("saillab/alpaca-slovak-cleaned", split=split)

    def preprocessor(self, example, add_prefix=True):
        instruction_input = example["input"]

        input_texts = [
            example["instruction"],
            instruction_input,
        ]

        if instruction_input == "nan":
            input_texts = [example["instruction"]]

        label_texts = [example["output"]]

        return self.formater(
            self.name,
            input_texts,
            label_texts,
            add_prefix,
            instruct=True,
            generation=True,
        )


TASK_MAPPING = OrderedDict(
    [
        ("sst2", SST2),
        ("qnli", QNLI),
        ("mnli", MNLI),
        ("yelp", YelpPolarity),
        ("trec", TREC),
        ("dbpedia", DBPEDIA),
        ("slovak_alpaca", SlovakAlpaca),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](seed=seed)

        raise ValueError(
            f"Unrecognized task {task} for AutoTask.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
