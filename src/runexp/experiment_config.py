from typing import List


class ExperimentConfig:
    def __init__(self, model_name: str, dataset_name: str, labeled_set_size: int, random_seed: int, cuda_device: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.labeled_set_size = labeled_set_size

        self.random_seed = random_seed
        self.cuda_device = cuda_device

    @staticmethod
    def deserialize(data: dict) -> "ExperimentConfig":
        return ExperimentConfig(
            model_name=data.get("model_name"),
            dataset_name=data.get("dataset_name"),
            labeled_set_size=data.get("labeled_set_size"),
            random_seed=data.get("random_seed"),
            cuda_device=data.get("cuda_device"),
        )
