#!/usr/bin/python

import argparse
from collections import defaultdict

import glob
import os

from runexp import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("--experiments-dir", help="Experiments directory", required=True)
    parser.add_argument("--experiment-name", help="Experiment name", required=False)

    parser.add_argument("--config-filename", default="config.json", required=False)
    parser.add_argument("--results-filename", default="eval_results.json", required=False)

    parser.add_argument("--key-fields", nargs="+", default=["experiment_name",
                                                            "model_name",
                                                            "dataset_name",
                                                            "self_learning_methods",
                                                            "labeled_train_set_size"])
    parser.add_argument("--agg-fields", nargs="+", default=["random_seed"])

    return parser.parse_args()


def main():
    cli_args = parse_args()

    experiments_dir = cli_args.experiments_dir
    experiment_name = cli_args.experiment_name

    config_filename = cli_args.config_filename
    results_filename = cli_args.results_filename

    key_fields = cli_args.key_fields
    agg_fields = cli_args.agg_fields

    configs_by_key = defaultdict(list)

    for directory in glob.glob("{}/{}*".format(experiments_dir, experiment_name + "_" if experiment_name else "")):
        config_path = os.path.join(directory, config_filename)
        results_path = os.path.join(directory, results_filename)
        if os.path.exists(config_path) and os.path.exists(results_path):
            config = ExperimentConfig.load_from_file(config_path)
            if experiment_name and config.experiment_name != experiment_name:
                continue

            key = tuple("{}={}".format(field[:2], getattr(config, field)) for field in key_fields)
            configs_by_key[key].append(config)

    for key, configs in configs_by_key.items():
        print("{}:".format(" ".join(key)))
        for config in configs:
            desc = tuple("{}={}".format(field[:2], getattr(config, field)) for field in agg_fields)
            print("\t{}".format(" ".join(desc)))


if __name__ == "__main__":
    main()
