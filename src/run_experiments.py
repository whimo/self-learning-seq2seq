import argparse
import json

from runexp import ExperimentConfig

import runexp.run_experiment as run_exp
import self_learning.domain_adaptation as da


class RunType:
    EXPERIMENT = "experiment"
    DOMAIN_ADAPTATION = "domain_adaptation"

    ALL = {
        EXPERIMENT,
        DOMAIN_ADAPTATION,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    parser.add_argument("-c", "--configs", help="Config files", nargs="+", required=True)
    parser.add_argument("--run-type", default=RunType.EXPERIMENT, choices=RunType.ALL)

    return parser.parse_args()


def main():
    cli_args = parse_args()

    config_files = cli_args.configs
    run_type = cli_args.run_type

    for config_file in config_files:
        config_data = json.load(open(config_file, "r"))

        config = ExperimentConfig.deserialize(config_data)

        if run_type == RunType.EXPERIMENT:
            run_exp.run_single_experiment(config)
        elif run_type == RunType.DOMAIN_ADAPTATION:
            da.run_domain_adaptation_training(config)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
