from typing import Optional
import math

import datasets
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

from runexp import ExperimentConfig
from models import ModelWrapper
from dataproc import DatasetWrapper
import dataproc.data_processing_helpers as data_help


def get_augmenter(augmentation_type: str, augmenter_kwargs: Optional[dict], device: Optional[str] = None):
    augmenter_kwargs = augmenter_kwargs or {}

    if augmentation_type == "synonym":
        return naw.SynonymAug(**augmenter_kwargs)

    if augmentation_type == "mlm":
        augmenter_kwargs["model_path"] = augmenter_kwargs.get("model_path", "roberta-base")
        augmenter_kwargs["model_type"] = augmenter_kwargs.get("model_type", "roberta")
        augmenter_kwargs["action"] = augmenter_kwargs.get("action", "substitute")
        augmenter_kwargs["device"] = device
        return naw.ContextualWordEmbsAug(**augmenter_kwargs)

    if augmentation_type == "emb":
        augmenter_kwargs["action"] = augmenter_kwargs.get("action", "substitute")
        return naw.WordEmbsAug(**augmenter_kwargs)

    raise Exception("Unkonwn augmentation type: {}".format(augmentation_type))


class AugmentedDatasetWrapper(DatasetWrapper):
    def __init__(self, **kwargs):
        super(AugmentedDatasetWrapper, self).__init__(**kwargs)

        self.aug_dataset = None
        self.preprocessed_aug_dataset = None

    def augment(self, augmentation_type: str, augmenter_kwargs: Optional[dict], random_seed: int,
                augmentation_scale: float = 1.0, device: Optional[str] = None):
        augmenter = get_augmenter(augmentation_type=augmentation_type, augmenter_kwargs=augmenter_kwargs, device=device)

        def augment_fn(data):
            data[self.input_field] = augmenter.augment(data[self.input_field])
            return data

        original_train_data = self.dataset[self.train_split_name]
        n_augs = math.ceil(augmentation_scale)
        aug_dataset_parts = []
        for i in range(n_augs):
            part = original_train_data.map(augment_fn, batched=True)
            aug_dataset_parts.append(part)

        aug_dataset = datasets.concatenate_datasets(aug_dataset_parts, axis=0)
        self.aug_dataset = data_help.get_random_sample_from_dataset(aug_dataset, seed=random_seed,
                                                                    size=math.ceil(len(original_train_data) * augmentation_scale))

    def preprocess_for_model(self, model: ModelWrapper):
        assert self.dataset
        assert self.aug_dataset

        preprocess_func = self.get_preprocess_func(tokenizer=model.tokenizer)
        self.preprocessed_dataset = self.dataset.map(preprocess_func, batched=True)
        self.preprocessed_aug_dataset = self.aug_dataset.map(preprocess_func, batched=True)

    @property
    def train_data(self):
        if not self.aug_dataset:
            raise Exception("No augmented dataset found, please call augment() method or use plain dataset wrapper")
        all_data = datasets.concatenate_datasets([self.dataset[self.train_split_name], self.aug_dataset], axis=0)
        return all_data.shuffle()

    @property
    def preprocessed_train_data(self):
        assert self.preprocessed_dataset and self.preprocessed_aug_dataset
        all_data = datasets.concatenate_datasets([self.preprocessed_dataset[self.train_split_name], self.preprocessed_aug_dataset], axis=0)
        return all_data.shuffle()
