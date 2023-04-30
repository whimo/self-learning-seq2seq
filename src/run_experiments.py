import argparse
import json

import runexp.run_experiment as run_exp
from runexp import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("-c", "--configs", help="Config files", nargs="+", required=True)

    return parser.parse_args()


def main():
    cli_args = parse_args()

    config_files = cli_args.configs

    for config_file in config_files:
        config_data = json.load(open(config_file, "r"))

        config = ExperimentConfig.deserialize(config_data)
        run_exp.run_single_experiment(config)


if __name__ == "__main__":
    main()
