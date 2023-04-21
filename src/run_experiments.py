import argparse
import json

import runexp.run_experiment as run_exp
from runexp import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("--config", help="Config file", required=True)

    return parser.parse_args()


def main():
    cli_args = parse_args()

    config_file = cli_args.config
    config_data = json.load(open(config_file, "r"))

    config = ExperimentConfig.deserialize(config_data)
    return run_exp.run_single_experiment(config)


if __name__ == "__main__":
    main()
