from typing import Optional
from typing import List
from typing import Dict

import pandas
import seaborn

import visualize.collect_data as collect_data


def draw_single_plot(data: pandas.DataFrame, x_column: str, target_column: str, hue_column: Optional[str],
                     title: Optional[str] = None, x_label: Optional[str] = None, y_label: Optional[str] = None):
    plot = seaborn.lineplot(data=data, x=x_column, y=target_column, hue=hue_column)
    if title:
        plot.set(title=title)
    if x_label:
        plot.set(x_label=x_label)
    if y_label:
        plot.set(y_label=y_label)
    return plot.get_figure()


def draw_multiple_plots(data: Dict[str, pandas.DataFrame], x_column: str, target_column: str,
                        title: Optional[str] = None, x_label: Optional[str] = None, y_label: Optional[str] = None):
    merged_df = pandas.concat([
        df.assign(label=label)
        for label, df in data.items()
    ]).reset_index()

    plot = seaborn.lineplot(data=merged_df, x=x_column, y=target_column, hue="label")
    if title:
        plot.set(title=title)
    if x_label:
        plot.set(x_label=x_label)
    if y_label:
        plot.set(y_label=y_label)
    return plot.get_figure()


def prepare_data_for_plots(experiments_dir: str,
                           key_columns: List[str], aggregate_columns: List[str],
                           experiment_name: Optional[str] = None, filter_duplicates: bool = True,
                           config_filename: str = "config.json", results_filename: str = "eval_results.json"):
    experiments_data = collect_data.collect_data_for_experiments(experiments_dir=experiments_dir, experiment_name=experiment_name,
                                                                 config_filename=config_filename, results_filename=results_filename)
    if filter_duplicates:
        experiments_data = collect_data.filter_duplicate_experiments(experiments_data, key_columns=key_columns,
                                                                     aggregate_columns=aggregate_columns)

    experiments_data_grouped = experiments_data.groupby(key_columns)
    data_by_experiment = {key: group for key, group in experiments_data_grouped}
    return data_by_experiment
