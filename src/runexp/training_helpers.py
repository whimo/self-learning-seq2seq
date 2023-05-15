import logging
import os
import glob
import shutil

from transformers import Seq2SeqTrainingArguments

from models import ModelWrapper
from dataproc import DatasetWrapper
from dataproc import DatasetName
from runexp import ExperimentConfig

import metrics.compute_metrics as cm


def get_training_args(config: ExperimentConfig):
    additional_args = config.additional_training_args or {}

    args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        include_inputs_for_metrics=True,
        load_best_model_at_end=True,
        metric_for_best_model=config.validation_metric,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
        warmup_ratio=config.warmup_ratio,
        save_total_limit=config.num_checkpoints_to_save,
        num_train_epochs=config.n_epochs,
        predict_with_generate=True,
        generation_num_beams=config.generation_num_beams,
        generation_max_length=config.max_target_length,
        fp16=config.do_use_gpu,
        **additional_args,
    )
    return args


def get_compute_metrics_fn(config: ExperimentConfig, model: ModelWrapper, dataset: DatasetWrapper, compute_additional_metrics: bool = False):
    additional_metrics = config.additional_metrics or [] if compute_additional_metrics else []

    if dataset.name == DatasetName.TRIVIA_QA:
        def compute_metrics(eval_preds):
            return cm.compute_metrics_for_qa(eval_preds=eval_preds,
                                             tokenizer=model.tokenizer,
                                             answers_by_question=dataset.qa_validation_answers_by_question)
        return compute_metrics

    def compute_metrics(eval_preds):
        return cm.compute_metrics(eval_preds=eval_preds,
                                  tokenizer=model.tokenizer,
                                  batch_size=config.eval_batch_size,
                                  add_metrics_to_use=additional_metrics)
    return compute_metrics


def delete_checkpoints(config: ExperimentConfig, checkpoints_prefix: str = "checkpoint"):
    assert config.output_dir

    for directory in glob.glob(os.path.join(config.output_dir, checkpoints_prefix) + "*"):
        logging.info("Deleting directory %s", directory)
        shutil.rmtree(directory)
