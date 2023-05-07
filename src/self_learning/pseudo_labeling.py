from typing import Optional

import logging
import copy

import numpy as np
from torch import nn

import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from runexp import ExperimentConfig
from models import ModelWrapper


class Seq2SeqTrainerForPseudoLabeling(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight", None)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))

        if weights is not None:
            weights = weights.repeat(labels.shape[1], 1).transpose(0, 1).contiguous()
            loss *= weights.view(-1)

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def get_pseudo_label_weighter(pseudo_labeling_args: Optional[dict]) -> Optional[callable]:
    pseudo_labeling_args = pseudo_labeling_args or {}
    weighter_type = pseudo_labeling_args.get("weighter_type")
    if weighter_type is None:
        return None

    if weighter_type == "linear-by-epoch":
        def weighter_inner(epoch, row):
            weight_coef = pseudo_labeling_args.get("weight_coef", 0)
            return weight_coef * epoch
        return weighter_inner

    if weighter_type == "const":
        def weighter_inner(epoch, row):
            weight = pseudo_labeling_args.get("weight", 1)
            return weight
        return weighter_inner

    raise Exception("Unknown weighter type: {}".format(weighter_type))


def get_pseudo_label_filter(pseudo_labeling_args: Optional[dict]) -> Optional[callable]:
    pseudo_labeling_args = pseudo_labeling_args or {}
    filter_type = pseudo_labeling_args.get("filter_type")
    if filter_type is None:
        return None

    if filter_type == "score-quantiles":
        def filter_inner(data):
            low_quantile = pseudo_labeling_args.get("low_quantile", 0.0)
            high_quantile = pseudo_labeling_args.get("high_quantile", 1.0)
            low_score = np.quantile(data["sequences_scores"], low_quantile)
            high_score = np.quantile(data["sequences_scores"], high_quantile)

            data = data.filter(lambda row: low_score < row["sequences_scores"] < high_score)
            return data
        return filter_inner

    raise Exception("Unknown filter type: {}".format(filter_type))


class ModelWrapperForPseudoLabeling(ModelWrapper):
    def __init__(self,
                 hf_checkpoint_path: str,
                 hf_model_class=AutoModelForSeq2SeqLM,
                 hf_tokenizer_class=AutoTokenizer,
                 hf_data_collator_class=DataCollatorForSeq2Seq,
                 hf_training_arguments_class=Seq2SeqTrainingArguments,
                 hf_trainer_class=Seq2SeqTrainerForPseudoLabeling):
        super(ModelWrapperForPseudoLabeling, self).__init__(
            hf_checkpoint_path=hf_checkpoint_path,
            hf_model_class=hf_model_class,
            hf_tokenizer_class=hf_tokenizer_class,
            hf_data_collator_class=hf_data_collator_class,
            hf_training_arguments_class=hf_training_arguments_class,
            hf_trainer_class=hf_trainer_class
        )

    def train_and_eval(self, training_arguments, train_data, validation_data,
                       compute_metrics_fn: callable, compute_metrics_fn_final: callable,
                       config: ExperimentConfig,
                       unlabeled_data=None, pseudo_label_weighter: callable = None, pseudo_label_filter: callable = None):
        assert isinstance(training_arguments, self.hf_training_arguments_class)
        training_arguments = copy.deepcopy(training_arguments)

        validation_data = self.remove_excess_columns(validation_data)

        data_collator = self.get_data_collator()
        train_data = train_data.map(lambda row: {"weight": 1.0})
        training_arguments.remove_unused_columns = False
        trainer = None
        pseudo_labeled_data = None

        if pseudo_label_weighter is None:
            pseudo_label_weighter = get_pseudo_label_weighter(config.self_learning_params.pseudo_labeling_params)
        if pseudo_label_filter is None:
            pseudo_label_filter = get_pseudo_label_filter(config.self_learning_params.pseudo_labeling_params)

        logging.info("Starting training with pseudo labeling")
        n_epochs = training_arguments.num_train_epochs
        for epoch in range(1, n_epochs + 1):
            logging.info("Epoch %s of %s", epoch, n_epochs)

            if epoch == 1:
                train_data_with_pseudo_labeled = train_data
            else:
                train_data_with_pseudo_labeled = datasets.concatenate_datasets([train_data, pseudo_labeled_data], axis=0)

            train_data_with_pseudo_labeled = self.remove_excess_columns(train_data_with_pseudo_labeled)

            logging.info("Training using existing data")
            training_arguments.num_train_epochs = 1
            trainer = self.hf_trainer_class(
                model=self.model,
                args=training_arguments,
                train_dataset=train_data_with_pseudo_labeled,
                eval_dataset=validation_data,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn
            )

            trainer.train()

            if epoch < n_epochs:
                logging.info("Generating pseudo labels")
                if "labels" in unlabeled_data.features:
                    unlabeled_data = unlabeled_data.remove_columns("labels")
                labels, sequences_scores = self.generate(data=unlabeled_data, config=config, max_length=training_arguments.generation_max_length)

                logging.info("Preparing generated labels")
                labels_dataset = datasets.Dataset.from_dict({"labels": labels, "sequences_scores": sequences_scores})
                pseudo_labeled_data = datasets.concatenate_datasets([unlabeled_data, labels_dataset], axis=1)

                logging.info("Weighting pseudo labels")
                pseudo_labeled_data = pseudo_labeled_data.map(
                    lambda row: {"weight": pseudo_label_weighter(epoch, row) if pseudo_label_weighter else 1.0}
                )
                if pseudo_label_filter:
                    logging.info("Filtering pseudo labels")
                    pseudo_labeled_data = pseudo_label_filter(pseudo_labeled_data)

        logging.info("Starting evaluation")
        trainer.compute_metrics = compute_metrics_fn_final
        return trainer.evaluate()
