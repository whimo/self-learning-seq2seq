import logging

import datasets
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from runexp import ExperimentConfig


class ModelWrapper:
    INPUT_COLUMNS_WHITELIST = {"input_ids", "decoder_input_ids", "attention_mask", "labels", "weight"}

    def __init__(self,
                 hf_checkpoint_path: str,
                 hf_model_class=AutoModelForSeq2SeqLM,
                 hf_tokenizer_class=AutoTokenizer,
                 hf_data_collator_class=DataCollatorForSeq2Seq,
                 hf_training_arguments_class=Seq2SeqTrainingArguments,
                 hf_trainer_class=Seq2SeqTrainer):
        self.model = None
        self.tokenizer = None

        self.hf_checkpoint_path = hf_checkpoint_path

        self.hf_model_class = hf_model_class
        self.hf_tokenizer_class = hf_tokenizer_class
        self.hf_data_collator_class = hf_data_collator_class
        self.hf_training_arguments_class = hf_training_arguments_class
        self.hf_trainer_class = hf_trainer_class

    def load_from_huggingface(self):
        self.model = self.hf_model_class.from_pretrained(self.hf_checkpoint_path)
        self.tokenizer = self.hf_tokenizer_class.from_pretrained(self.hf_checkpoint_path)

    @classmethod
    def construct_with_config(cls, config: ExperimentConfig) -> "ModelWrapper":
        if config.model_name.startswith("bart-"):
            hf_checkpoint_path = "facebook/{}".format(config.model_name)
        elif config.model_name.startswith("t5-"):
            hf_checkpoint_path = config.model_name
        else:
            raise NotImplementedError("Unsupported model: {}".format(config.model_name))

        logging.info("Constructing model from Huggingface checkpoint %s", hf_checkpoint_path)

        model = cls(
            hf_checkpoint_path=hf_checkpoint_path,
        )
        model.load_from_huggingface()
        return model

    def get_data_collator(self):
        return self.hf_data_collator_class(tokenizer=self.tokenizer, model=self.model,
                                           padding=True, label_pad_token_id=self.tokenizer.pad_token_id)

    @classmethod
    def remove_excess_columns(cls, data: datasets.Dataset):
        return data.remove_columns([column for column in data.features.keys()
                                    if column not in cls.INPUT_COLUMNS_WHITELIST])

    def train_and_eval(self, training_arguments, train_data, validation_data, compute_metrics_fn, compute_metrics_fn_final,
                       config: ExperimentConfig, **kwargs):
        assert isinstance(training_arguments, self.hf_training_arguments_class)

        data_collator = self.get_data_collator()

        trainer = self.hf_trainer_class(
            self.model,
            training_arguments,
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn
        )

        logging.info("Starting training")
        trainer.train()

        logging.info("Starting evaluation")
        trainer.compute_metrics = compute_metrics_fn_final
        return trainer.evaluate()

    def generate(self, data, config: ExperimentConfig, **kwargs):
        prepared_data = data.remove_columns([column for column in data.features.keys()
                                             if column not in ModelWrapper.INPUT_COLUMNS_WHITELIST])
        dataloader = DataLoader(
            prepared_data,
            batch_size=config.eval_batch_size,
            collate_fn=self.get_data_collator()
        )

        device = self.model.device
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        total_sequences = num_return_sequences * len(data)
        sequences = (
                torch.zeros(
                    (total_sequences, config.max_target_length), dtype=torch.int64, device=device
                )
                + self.tokenizer.pad_token_id
        )
        scores = torch.empty(total_sequences, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            start = 0
            for batch in tqdm(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model.generate(
                    **batch,
                    max_length=config.max_target_length,
                    min_length=3,  # To avoid empty outputs. 3 == <BOS> + at least one token + <EOS>
                    output_scores=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
                end = start + (len(batch["input_ids"]) * num_return_sequences)
                sequences[start:end, : output.sequences.shape[1]].copy_(
                    output.sequences, non_blocking=True
                )
                scores[start:end].copy_(output.sequences_scores, non_blocking=True)
                start = end

        return sequences, scores
