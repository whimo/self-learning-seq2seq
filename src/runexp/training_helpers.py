from transformers import Seq2SeqTrainingArguments

from models import ModelWrapper
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
        per_device_eval_batch_size=config.batch_size,
        include_inputs_for_metrics=True,
        load_best_model_at_end=True,
        metric_for_best_model=config.validation_metric,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        save_total_limit=config.num_checkpoints_to_save,
        num_train_epochs=config.n_epochs,
        predict_with_generate=True,
        generation_num_beams=config.generation_num_beams,
        fp16=config.do_use_gpu,
        **additional_args,
    )
    return args


def get_compute_metrics_fn(config: ExperimentConfig, model: ModelWrapper):
    def compute_metrics(eval_preds):
        return cm.compute_metrics(eval_preds=eval_preds,
                                  tokenizer=model.tokenizer,
                                  batch_size=config.batch_size,
                                  add_metrics_to_use=config.additional_metrics or [])
    return compute_metrics
