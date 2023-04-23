import logging

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from runexp import ExperimentConfig


class ModelWrapper:
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

    @staticmethod
    def construct_with_config(config: ExperimentConfig) -> "ModelWrapper":
        if config.model_name.startswith("bart-"):
            hf_checkpoint_path = "facebook/{}".format(config.model_name)
        elif config.model_name.startswith("t5-"):
            hf_checkpoint_path = config.model_name
        else:
            raise NotImplementedError("Unsupported model: {}".format(config.model_name))

        logging.info("Constructing model from Huggingface checkpoint %s", hf_checkpoint_path)

        model = ModelWrapper(
            hf_checkpoint_path=hf_checkpoint_path,
        )
        model.load_from_huggingface()
        return model

    def train_and_eval(self, training_arguments, train_data, validation_data, compute_metrics_fn, compute_metrics_fn_final):
        assert isinstance(training_arguments, self.hf_training_arguments_class)

        data_collator = self.hf_data_collator_class(tokenizer=self.tokenizer, model=self.model,
                                                    padding=True, label_pad_token_id=self.tokenizer.pad_token_id)

        trainer = self.hf_trainer_class(
            self.model,
            training_arguments,
            train_dataset=train_data,
            eval_dataset=validation_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn
        )

        trainer.che

        logging.info("Starting training")
        trainer.train()

        logging.info("Starting evaluation")
        trainer.compute_metrics = compute_metrics_fn_final
        return trainer.evaluate()
