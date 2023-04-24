from typing import Optional
from typing import List

import os
import glob
import json

import pandas

from runexp import ExperimentConfig


def collect_data_for_experiments(experiments_dir: str, experiment_name: Optional[str] = None,
                                 config_filename: str = "config.json", results_filename: str = "eval_results.json") -> pandas.DataFrame:
    experiments_data_rows = []

    dir_expression = "{}/{}*".format(experiments_dir, experiment_name + "_" if experiment_name else "")
    for directory in glob.glob(dir_expression):
        config_path = os.path.join(directory, config_filename)
        results_path = os.path.join(directory, results_filename)
        if os.path.exists(config_path) and os.path.exists(results_path):
            config = ExperimentConfig.load_from_file(config_path)
            if experiment_name and config.experiment_name != experiment_name:
                continue

            exp_data = config.serialize()
            with open(results_path, "r") as fd:
                results_data = json.load(fd)
                exp_data.update(results_data)

            experiments_data_rows.append(exp_data)

    experiments_data = pandas.json_normalize(experiments_data_rows)
    return experiments_data


def filter_duplicate_experiments(experiments_data: pd.DataFrame, key_columns: List[str], aggregate_columns: List[str]):
    return experiments_data[~experiments_data.duplicated(key_columns + aggregate_columns)]
