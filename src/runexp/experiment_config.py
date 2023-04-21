from typing import Optional
import json


class DefaultValues:
    RANDOM_SEED = 0
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 64

    BATCH_SIZE = 16
    N_EPOCHS = 1
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1
    WARMUP_RATIO = 0
    GENERATION_NUM_BEAMS = 4

    NUM_CHECKPOINTS_TO_SAVE = 2


class ExperimentConfig:
    def __init__(self,
                 model_name: str, dataset_name: str,
                 labeled_train_set_size: Optional[int] = None,
                 validation_set_size: Optional[int] = None,
                 random_seed: int = DefaultValues.RANDOM_SEED,
                 cuda_device: Optional[str] = None,
                 max_input_length: int = DefaultValues.MAX_INPUT_LENGTH,
                 max_target_length: int = DefaultValues.MAX_TARGET_LENGTH,
                 batch_size: int = DefaultValues.BATCH_SIZE,
                 n_epochs: int = DefaultValues.N_EPOCHS,
                 learning_rate: float = DefaultValues.LEARNING_RATE,
                 weight_decay: float = DefaultValues.WEIGHT_DECAY,
                 max_grad_norm: float = DefaultValues.MAX_GRAD_NORM,
                 warmup_ratio: float = DefaultValues.WARMUP_RATIO,
                 generation_num_beams: int = DefaultValues.GENERATION_NUM_BEAMS,
                 num_checkpoints_to_save: int = DefaultValues.NUM_CHECKPOINTS_TO_SAVE,
                 additional_training_args: Optional[dict] = None,
                 additional_metrics: Optional[list] = None,
                 output_dir: Optional[str] = None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.labeled_train_set_size = labeled_train_set_size
        self.validation_set_size = validation_set_size

        self.random_seed = random_seed
        self.cuda_device = cuda_device

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.generation_num_beams = generation_num_beams
        self.num_checkpoints_to_save = num_checkpoints_to_save

        self.additional_training_args = additional_training_args
        self.additional_metrics = additional_metrics

        self.output_dir = output_dir

    @property
    def do_use_gpu(self):
        return self.cuda_device is not None

    def __repr__(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.random_seed)

    @staticmethod
    def deserialize(data: dict) -> "ExperimentConfig":
        return ExperimentConfig(
            model_name=data.get("model_name"),
            dataset_name=data.get("dataset_name"),
            labeled_train_set_size=data.get("labeled_train_set_size"),
            validation_set_size=data.get("validation_set_size"),
            random_seed=data.get("random_seed", DefaultValues.RANDOM_SEED),
            cuda_device=data.get("cuda_device"),
            max_input_length=data.get("max_input_length", DefaultValues.MAX_INPUT_LENGTH),
            max_target_length=data.get("max_target_length", DefaultValues.MAX_TARGET_LENGTH),
            batch_size=data.get("batch_size", DefaultValues.BATCH_SIZE),
            n_epochs=data.get("n_epochs", DefaultValues.N_EPOCHS),
            learning_rate=data.get("learning_rate", DefaultValues.LEARNING_RATE),
            weight_decay=data.get("weight_decay", DefaultValues.WEIGHT_DECAY),
            max_grad_norm=data.get("max_grad_norm", DefaultValues.MAX_GRAD_NORM),
            warmup_ratio=data.get("warmup_ratio", DefaultValues.WARMUP_RATIO),
            generation_num_beams=data.get("generation_num_beams", DefaultValues.GENERATION_NUM_BEAMS),
            num_checkpoints_to_save=data.get("num_checkpoints_to_save", DefaultValues.NUM_CHECKPOINTS_TO_SAVE),
            additional_training_args=data.get("additional_training_args"),
            additional_metrics=data.get("additional_metrics"),
            output_dir=data.get("output_dir"),
        )

    def serialize(self):
        data = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "labeled_train_set_size": self.labeled_train_set_size,
            "validation_set_size": self.validation_set_size,
            "random_seed": self.random_seed,
            "cuda_device": self.cuda_device,
            "max_input_length": self.max_input_length,
            "max_target_length": self.max_target_length,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "warmup_ratio": self.warmup_ratio,
            "generation_num_beams": self.generation_num_beams,
            "num_checkpoints_to_save": self.num_checkpoints_to_save,
            "additional_training_args": self.additional_training_args,
            "additional_metrics": self.additional_metrics,
            "output_dir": self.output_dir,
        }
        return data

    def dump_to_file(self, file_path: str):
        data = self.serialize()
        with open(file_path, "w+") as fd:
            json.dump(data, fd)