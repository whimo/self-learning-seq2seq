from typing import Optional

import copy
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
                 train_split_name: str = "train", validation_split_name: str = "validation", test_split_name: str = "test",
                 unlabeled_train_split_name: str = "unlabeled",
                 max_input_length: Optional[int] = None, max_target_length: Optional[int] = None):
        self.name = name

        self.hf_path = hf_path
        self.hf_config_name = hf_config_name

        self.input_field = input_field
        self.target_field = target_field

        self.train_split_name = train_split_name
        self.validation_split_name = validation_split_name
        self.test_split_name = test_split_name
        self.unlabeled_train_split_name = unlabeled_train_split_name

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.dataset = None
        self.preprocessed_dataset = None

    def load_from_huggingface(self):
        self.dataset = datasets.load_dataset(path=self.hf_path, name=self.hf_config_name)

    @classmethod
    def construct_with_config(cls, config: ExperimentConfig):
        if config.dataset_name == DatasetName.XSUM:
            hf_path = "xsum"
            hf_config_name = None
            input_field = "document"
            target_field = "summary"
            max_target_length = 48
        elif config.dataset_name == DatasetName.AESLC:
            hf_path = "aeslc"
            hf_config_name = None
            input_field = "email_body"
            target_field = "subject_line"
            max_target_length = 26
        elif config.dataset_name == DatasetName.PARABANK:
            hf_path = None
            hf_config_name = None
            input_field = "input"
            target_field = "output"
            max_target_length = 64
        elif config.dataset_name == DatasetName.QUORA:
            hf_path = "quora"
            hf_config_name = None
            input_field = "input"
            target_field = "output"
            max_target_length = 32
        elif config.dataset_name == DatasetName.ELI5:
            hf_path = "eli5"
            hf_config_name = None
            input_field = "input"
            target_field = "output"
            max_target_length = 840
        elif config.dataset_name == DatasetName.TRIVIA_QA:
            hf_path = "trivia_qa"
            hf_config_name = "unfiltered.nocontext"
            input_field = "input"
            target_field = "output"
            max_target_length = 16
        else:
            raise NotImplementedError

        dataset = cls(
            name=config.dataset_name,
            hf_path=hf_path,
            hf_config_name=hf_config_name,
            input_field=input_field,
            target_field=target_field,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length or max_target_length,
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
            dataset.train_split_name = "train_eli5"
            dataset.validation_split_name = "validation_eli5"

        elif config.dataset_name == DatasetName.TRIVIA_QA:
            dataset.load_from_huggingface()

            def prepare(row):
                return {"input": row["question"], "output": row["answer"]["value"]}

            dataset.dataset = dataset.dataset.map(prepare)
        else:
            dataset.load_from_huggingface()

        return dataset

    def get_preprocess_func(self, tokenizer,
                            labels_field: str = "labels", tokenizer_output_field: str = "input_ids",
                            truncation: bool = True):
        def preprocess(data):
            model_inputs = tokenizer(data[self.input_field], max_length=self.max_input_length, truncation=truncation)

            if self.target_field in data:
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(data[self.target_field], max_length=self.max_target_length, truncation=truncation)
                model_inputs[labels_field] = labels[tokenizer_output_field]

            return model_inputs

        return preprocess

    def preprocess_for_model(self, model: ModelWrapper):
        assert self.dataset

        preprocess_func = self.get_preprocess_func(tokenizer=model.tokenizer)
        self.preprocessed_dataset = self.dataset.map(preprocess_func, batched=True)

    def get_random_train_data_subset(self, size: int, seed: int, preprocessed: bool = True):
        if preprocessed:
            assert self.preprocessed_dataset
            dataset = self.preprocessed_dataset
        else:
            dataset = self.dataset
        return data_help.get_random_sample_from_dataset(dataset=dataset, size=size, seed=seed)

    def get_random_validation_data_subset(self, size: int, seed: int, preprocessed: bool = True):
        if preprocessed:
            assert self.preprocessed_dataset
            dataset = self.preprocessed_dataset
        else:
            dataset = self.dataset
        return data_help.get_random_sample_from_dataset(dataset=dataset, size=size, seed=seed)

    def get_random_labeled_and_unlabeled_train_data(self, labeled_dataset_size: int, seed: int, unlabeled_dataset_size: Optional[int] = None,
                                                    labels_field: Optional[str] = "labels", preprocessed: bool = True):
        if preprocessed:
            assert self.preprocessed_dataset
            dataset = self.preprocessed_dataset
        else:
            dataset = self.dataset
        labeled_data, unlabeled_data = data_help.split_dataset_into_two_parts_randomly(dataset=dataset,
                                                                                       first_part_size=labeled_dataset_size,
                                                                                       second_part_size=unlabeled_dataset_size,
                                                                                       seed=seed)
        if self.target_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(self.target_field)
        if labels_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(labels_field)
        return labeled_data, unlabeled_data

    def split_train_data_into_labeled_and_unlabeled(self, labeled_dataset_size: int, seed: int, unlabeled_dataset_size: Optional[int] = None,
                                                    labels_field: Optional[str] = "labels"):
        labeled_data_indices, unlabeled_data_indices = data_help.get_random_split_indices(
            length=len(self.dataset[self.train_split_name]),
            first_part_size=labeled_dataset_size,
            second_part_size=unlabeled_dataset_size,
            seed=seed
        )

        unlabeled_data = self.dataset[self.train_split_name].select(unlabeled_data_indices)
        labeled_data = self.dataset[self.train_split_name].select(labeled_data_indices)

        if self.target_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(self.target_field)
        if labels_field in unlabeled_data.features:
            unlabeled_data = unlabeled_data.remove_columns(labels_field)
        self.dataset[self.train_split_name] = labeled_data
        self.dataset[self.unlabeled_train_split_name] = unlabeled_data

        if self.preprocessed_dataset:
            preprocessed_unlabeled_data = self.train_data.select(unlabeled_data_indices)
            preprocessed_labeled_data = self.train_data.select(labeled_data_indices)

            if self.target_field in preprocessed_unlabeled_data.features:
                preprocessed_unlabeled_data = preprocessed_unlabeled_data.remove_columns(self.target_field)
            if labels_field in preprocessed_unlabeled_data.features:
                preprocessed_unlabeled_data = preprocessed_unlabeled_data.remove_columns(labels_field)

            self.preprocessed_dataset[self.train_split_name] = preprocessed_labeled_data
            self.preprocessed_dataset[self.unlabeled_train_split_name] = preprocessed_unlabeled_data

    def clone_without_data(self):
        return DatasetWrapper(
            name=self.name,
            hf_path=None,
            hf_config_name=None,
            input_field=self.input_field,
            target_field=self.target_field,
            train_split_name=self.train_split_name,
            validation_split_name=self.validation_split_name,
            test_split_name=self.test_split_name,
            max_input_length=self.max_input_length,
            max_target_length=self.max_target_length
        )

    def clone_with_random_train_subset(self, size: int, seed: int) -> "DatasetWrapper":
        new_dataset_wrapper = self.clone_without_data()

        train_indices = data_help.get_random_sample_indices(length=len(self.dataset[self.train_split_name]), size=size, seed=seed)

        train_subset = self.train_data.select(train_indices)
        new_dataset = copy.deepcopy(self.dataset)
        new_dataset[self.train_split_name] = train_subset
        new_dataset_wrapper.dataset = new_dataset

        if self.preprocessed_dataset:
            preprocesseed_train_subset = self.train_data.select(train_indices)
            preprocessed_new_dataset = copy.deepcopy(self.preprocessed_dataset)
            preprocessed_new_dataset[self.train_split_name] = preprocesseed_train_subset
            new_dataset_wrapper.preprocessed_dataset = preprocessed_new_dataset

        return new_dataset_wrapper

    @property
    def train_data(self):
        return self.dataset[self.train_split_name]

    @property
    def unlabeled_train_data(self):
        return self.dataset[self.unlabeled_train_split_name]

    @property
    def validation_data(self):
        return self.dataset[self.validation_split_name]

    @property
    def test_data(self):
        return self.dataset[self.test_split_name]

    @property
    def preprocessed_train_data(self):
        if not self.preprocessed_dataset:
            return None
        return self.preprocessed_dataset[self.train_split_name]

    @property
    def preprocessed_unlabeled_train_data(self):
        if not self.preprocessed_dataset:
            return None
        return self.preprocessed_dataset[self.unlabeled_train_split_name]

    @property
    def preprocessed_validation_data(self):
        if not self.preprocessed_dataset:
            return None
        return self.preprocessed_dataset[self.validation_split_name]

    @property
    def preprocessed_test_data(self):
        if not self.preprocessed_dataset:
            return None
        return self.preprocessed_dataset[self.test_split_name]
