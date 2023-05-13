import logging
import os
from typing import Optional
from typing import List

import datasets
import numpy as np

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import EarlyStoppingCallback

import metrics.compute_metrics as cm

from models import ModelWrapper
from dataproc import DatasetWrapper

from self_learning import DataCollatorForDenoisingTasks
from self_learning import SentenceTokenize


def get_domain_adaptation_training_arguments(output_dir: str, generation_max_length: int, validation_metric: str):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=64,
        # Optimizer args
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.,
        warmup_ratio=0.1,
        metric_for_best_model=validation_metric,
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        # WANDB args
        report_to="none",
        # Seq2Seq args
        generation_max_length=generation_max_length,
        predict_with_generate=True,
        generation_num_beams=4,
        # General args
        fp16=True,
        # fp16_full_eval=False,
        include_inputs_for_metrics=True,
    )


def get_max_text_length(dataset, quantile: float = 0.99, input_field: str = "input_ids"):
    return np.quantile([len(text) for text in dataset[input_field]], quantile)


def run_domain_adaptation_training(model: ModelWrapper, dataset: DatasetWrapper,
                                   validation_set_size: int, validation_metric: str,
                                   output_dir: str = "domain_adaptation_output",
                                   random_seed: int = 0,
                                   compute_metrics_batch_size: int = 32,
                                   additional_metrics: Optional[List[str]] = None):
    logging.info("Preparing data")
    train_data = dataset.train_data
    if dataset.unlabeled_train_split_name in dataset.dataset:
        train_data = datasets.concatenate_datasets([train_data, dataset.unlabeled_train_data], axis=0)

    validation_data = dataset.get_random_validation_data_subset(size=validation_set_size, seed=random_seed)

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

    data_collator = DataCollatorForDenoisingTasks(tokenizer=model.tokenizer)

    def compute_metrics(eval_preds):
        return cm.compute_metrics(eval_preds=eval_preds,
                                  tokenizer=model.tokenizer,
                                  batch_size=compute_metrics_batch_size,
                                  add_metrics_to_use=additional_metrics or [])

    training_arguments = get_domain_adaptation_training_arguments(output_dir=output_dir, generation_max_length=generation_max_length,
                                                                  validation_metric=validation_metric)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = Seq2SeqTrainer(
        model_init=model.model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_validation_data,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    logging.info("Starting training")
    trainer.train()

    logging.info("Saving model")
    trainer.save_model(output_dir=os.path.join(output_dir, "final"))
