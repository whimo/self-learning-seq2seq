import argparse

import json

from dataproc import DatasetWrapper
from models import ModelWrapper

import models.training_manage as tm

from runexp import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("--config", help="Config file", required=True)

    return parser.parse_args()


def run_experiments(config: ExperimentConfig):
    model = ModelWrapper.construct_with_config(config)
    dataset = DatasetWrapper.construct_with_config(config)
    dataset.preprocess_for_model(model, max_input_length=1026, max_target_length=256)

    training_args = tm.get_training_args()

    model.train_and_eval(train_data=dataset.preprocessed_dataset["train"],
                         validation_data=dataset.preprocessed_dataset["validation"],
                         training_arguments=training_args)


def main():
    cli_args = parse_args()

    config_file = cli_args.config
    config_data = json.load(open(config_file, "r"))

    config = ExperimentConfig.deserialize(config_data)
    return run_experiments(config)


if __name__ == "__main__":
    main()
