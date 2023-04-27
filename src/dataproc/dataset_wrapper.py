from typing import Optional

import pandas as pd
import datasets
from datasets import Dataset

from runexp import ExperimentConfig
from models import ModelWrapper

import dataproc.data_processing_helpers as data_help

PATH_TO_PARABANK = "parabank/parabank.5m.tsv"
PARABANK_SAMPLE_SIZE = 100000


class DatasetName:
    XSUM = "xsum"
    AESLC = "aeslc"
    TRIVIA_QA = "trivia_qa"
    ELI5 = "eli5"
    PARABANK = "parabank"
    QUORA = "quora"


class DatasetWrapper:
    def __init__(self, name: str, hf_path: Optional[str], hf_config_name: Optional[str], input_field: str, target_field: str,
                 train_split_name: str = "train", validation_split_name: str = "validation", test_split_name: str = "test"):
        self.name = name

        self.hf_path = hf_path
        self.hf_config_name = hf_config_name

        self.input_field = input_field
        self.target_field = target_field

        self.train_split_name = train_split_name
        self.validation_split_name = validation_split_name
        self.test_split_name = test_split_name

        self.dataset = None
        self.preprocessed_dataset = None

    def load_from_huggingface(self):
        self.dataset = datasets.load_dataset(path=self.hf_path, name=self.hf_config_name)

    @staticmethod
    def construct_with_config(config: ExperimentConfig) -> "DatasetWrapper":
        if config.dataset_name == DatasetName.XSUM:
            hf_path = "xsum"
            hf_config_name = None
            input_field = "document"
            target_field = "summary"
        elif config.dataset_name == DatasetName.AESLC:
            hf_path = "aeslc"
            hf_config_name = None
            input_field = "email_body"
            target_field = "subject_line"
        elif config.dataset_name == DatasetName.PARABANK:
            hf_path = None
            hf_config_name = None
            input_field = "input"
            target_field = "output"
        elif config.dataset_name == DatasetName.QUORA:
            hf_path = "quora"
            hf_config_name = None
            input_field = "input"
            target_field = "output"
        elif config.dataset_name == DatasetName.ELI5:
            hf_path = "eli5"
            hf_config_name = None
            input_field = "input"
            target_field = "output"
        else:
            raise NotImplementedError

        dataset = DatasetWrapper(
            name=config.dataset_name,
            hf_path=hf_path,
            hf_config_name=hf_config_name,
            input_field=input_field,
            target_field=target_field,
        )

        if config.dataset_name == DatasetName.PARABANK:
            try:
                data = pd.read_csv(PATH_TO_PARABANK, sep="\t", on_bad_lines="skip").dropna().sample(PARABANK_SAMPLE_SIZE)
            except Exception as exc:
                raise Exception("Failed to load Parabank: {}".format(exc))
            data.columns = ["input", "output"]
            dataset.dataset = Dataset.from_pandas(data)
            dataset.dataset = dataset.dataset.train_test_split(test_size=0.2, shuffle=True)
            dataset.validation_split_name = "test"

        elif config.dataset_name == DatasetName.QUORA:
            dataset.load_from_huggingface()

            def expand(row):
                return {"input": row["questions"]["text"][0], "output": row["questions"]["text"][1]}

            dataset.dataset = dataset.dataset.filter(lambda row: row["is_duplicate"])
            dataset.dataset = dataset.dataset.map(expand)
            dataset.dataset = dataset.dataset["train"].train_test_split(test_size=0.2, shuffle=True)
            dataset.validation_split_name = "test"
        elif config.dataset_name == DatasetName.ELI5:
            dataset.load_from_huggingface()

            def prepare(row):
                return {"input": row["title"], "output": row["answers"]["text"][0]}

            dataset.dataset = dataset.dataset.filter(lambda row: len(row["answers"].get("text", [])) > 0)
            dataset.dataset = dataset.dataset.map(prepare)
            dataset.dataset = dataset.dataset["train_eli5"].train_test_split(test_size=0.2, shuffle=True)
            dataset.validation_split_name = "test"
        else:
            dataset.load_from_huggingface()

        return dataset

    def get_preprocess_func(self, tokenizer, max_input_length, max_target_length,
                            labels_field: str = "labels", tokenizer_output_field: str = "input_ids",
                            truncation: bool = True):
        def preprocess(data):
            model_inputs = tokenizer(data[self.input_field], max_length=max_input_length, truncation=truncation)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(data[self.target_field], max_length=max_target_length, truncation=truncation)

            model_inputs[labels_field] = labels[tokenizer_output_field]
            return model_inputs

        return preprocess

    def preprocess_for_model(self, model: ModelWrapper, max_input_length, max_target_length):
        assert self.dataset

        preprocess_func = self.get_preprocess_func(tokenizer=model.tokenizer,
                                                   max_input_length=max_input_length,
                                                   max_target_length=max_target_length)
        self.preprocessed_dataset = self.dataset.map(preprocess_func, batched=True)

    def get_random_train_data_subset(self, size: int, seed: int):
        assert self.preprocessed_dataset
        return data_help.get_random_sample_from_dataset(dataset=self.train_data, size=size, seed=seed)

    def get_random_validation_data_subset(self, size: int, seed: int):
        assert self.preprocessed_dataset
        return data_help.get_random_sample_from_dataset(dataset=self.validation_data, size=size, seed=seed)

    def get_random_labeled_and_unlabeled_train_data(self, labeled_dataset_size: int, seed: int, unlabeled_dataset_size: Optional[int] = None,
                                                    labels_field: Optional[str] = "labels"):
        assert self.preprocessed_dataset
        labeled_data, unlabeled_data = data_help.split_dataset_into_two_parts_randomly(dataset=self.train_data,
                                                                                       first_part_size=labeled_dataset_size,
                                                                                       second_part_size=unlabeled_dataset_size,
                                                                                       seed=seed)
        if self.target_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(self.target_field)
        if labels_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(labels_field)
        return labeled_data, unlabeled_data

    @property
    def train_data(self):
        assert self.preprocessed_dataset
        return self.preprocessed_dataset[self.train_split_name]

    @property
    def validation_data(self):
        assert self.preprocessed_dataset
        return self.preprocessed_dataset[self.validation_split_name]

    @property
    def test_data(self):
        assert self.preprocessed_dataset
        return self.preprocessed_dataset[self.test_split_name]
