import logging
import os

import numpy as np

import datasets
import transformers
from transformers import Seq2SeqTrainer
from transformers import EarlyStoppingCallback

from runexp import ExperimentConfig
from models import ModelWrapper
from dataproc import DatasetWrapper

import runexp.training_helpers as train_help
import dataproc.data_processing_helpers as data_help

from self_learning import DataCollatorForDenoisingTasks
from self_learning import SentenceTokenize


def get_max_text_length(dataset, quantile: float = 0.99, input_field: str = "input_ids"):
    return np.quantile([len(text) for text in dataset[input_field]], quantile)


def prepare_domain_adaptation_context(config: ExperimentConfig):
    logging.basicConfig(format='[%(levelname)s:%(process)d] %(asctime)s - %(message)s', level=logging.INFO)
    transformers.set_seed(config.random_seed)
    np.random.seed(config.random_seed)

    logging.info("Preparing model and dataset")
    model = ModelWrapper.construct_with_config(config)
    dataset = DatasetWrapper.construct_with_config(config)

    logging.info("Preparing data for domain adaptation")
    train_data = dataset.train_data
    if dataset.unlabeled_train_split_name in dataset.dataset:
        train_data = datasets.concatenate_datasets([train_data, dataset.unlabeled_train_data], axis=0)

    validation_data = dataset.get_random_validation_data_subset(size=config.validation_set_size, seed=config.random_seed)
    if config.self_learning_params.unlabeled_dataset_size:
        train_data = data_help.get_random_sample_from_dataset(train_data, size=config.self_learning_params.unlabeled_dataset_size,
                                                              seed=config.random_seed)

    train_data = train_data.rename_column(dataset.input_field, "text")
    validation_data = validation_data.rename_column(dataset.input_field, "text")

    sent_tok = SentenceTokenize()

    sent_tok_train_data = train_data.map(
        sent_tok, batched=True, remove_columns=train_data.column_names
    )
    sent_tok_validation_data = validation_data.map(
        sent_tok, batched=True, remove_columns=validation_data.column_names
    )

    def tokenize_fn(data):
        encoded = model.tokenizer(data["text"], truncation=True)
        return encoded

    tokenized_train_data = sent_tok_train_data.map(tokenize_fn, batched=True)
    tokenized_validation_data = sent_tok_validation_data.map(tokenize_fn, batched=True)

    generation_max_length = get_max_text_length(tokenized_train_data)
    logging.info("Generation max length (inferred from the dataset): %s", generation_max_length)

    if config.max_target_length is None:
        config.max_target_length = get_max_text_length(dataset=tokenized_train_data)
        logging.info("Max target length is not set in config, using value inferred from dataset: %s", config.max_target_length)

    logging.info("Preparing training params")
    training_args = train_help.get_training_args(config=config)

    return model, dataset, training_args, tokenized_train_data, tokenized_validation_data


def run_domain_adaptation_training(config: ExperimentConfig):
    logging.info("Preparing domain adaptation context")
    model, dataset, training_args, tokenized_train_data, tokenized_validation_data = prepare_domain_adaptation_context(config=config)

    data_collator = DataCollatorForDenoisingTasks(tokenizer=model.tokenizer)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    compute_metrics_fn = train_help.get_compute_metrics_fn(config=config, model=model, compute_additional_metrics=False)

    trainer = Seq2SeqTrainer(
        model_init=model.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_validation_data,
        callbacks=callbacks,
        compute_metrics=compute_metrics_fn
    )

    logging.info("Starting training")
    trainer.train()

    logging.info("Saving model")
    trainer.save_model(output_dir=os.path.join(config.output_dir, "final"))
