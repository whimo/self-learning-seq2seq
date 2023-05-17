from typing import Optional
from typing import List
from typing import Dict

import numpy as np
import pandas
import seaborn

import visualize.collect_data as collect_data


def draw_single_plot(data: pandas.DataFrame, x_column: str, target_column: str, hue_column: Optional[str],
                     title: Optional[str] = None, x_label: Optional[str] = None, y_label: Optional[str] = None):
    plot = seaborn.lineplot(data=data, x=x_column, y=target_column, hue=hue_column)
    if title:
        plot.set(title=title)
    if x_label:
        plot.set(xlabel=x_label)
    if y_label:
        plot.set(ylabel=y_label)
    return plot.get_figure()


def draw_multiline_plot(data: Dict[str, pandas.DataFrame], x_column: str, target_column: str,
                        legend_title: str, x_scale: str = "linear",
                        title: Optional[str] = None, x_label: Optional[str] = None, y_label: Optional[str] = None):
    merged_df = pandas.concat([
        df.assign(**{legend_title: label})
        for label, df in data.items()
    ]).reset_index()

    plot = seaborn.lineplot(data=merged_df, x=x_column, y=target_column, hue=legend_title, style=legend_title)
    x_ticks = list(merged_df[x_column].unique())
    if title:
        plot.set(title=title)
    if x_label:
        plot.set(xlabel=x_label)
    if y_label:
        plot.set(ylabel=y_label)
    
    plot.set(xscale=x_scale)
    plot.set(xticks=x_ticks)
    plot.set(xticklabels=x_ticks)
    return plot.get_figure()


def draw_multiline_plots_grid(data: Dict[str, pandas.DataFrame], x_column: str, target_column: str,
                              grid_column: str, legend_title: str, x_scale: str = "linear",
                              title: Optional[str] = None, x_label: Optional[str] = None, y_label: Optional[str] = None):
    merged_df = pandas.concat([
        df.assign(**{legend_title: label})
        for label, df in data.items()
    ]).reset_index()

    plot = seaborn.relplot(data=merged_df, x=x_column, y=target_column, col=grid_column,
                           hue=legend_title, style=legend_title, kind="line")
    x_ticks = list(merged_df[x_column].unique())
    if title:
        plot.set(title=title)
    if x_label:
        plot.set(xlabel=x_label)
    if y_label:
        plot.set(ylabel=y_label)
    
    plot.set(xscale=x_scale)
    plot.set(xticks=x_ticks)
    plot.set(xticklabels=x_ticks)
    return plot

def normalize_experiment_key(key):
    return tuple(item if not pandas.isna(item) else None for item in key)


def prepare_data_for_plots(experiments_dir: str,
                           key_columns: List[str], aggregate_columns: List[str],
                           experiment_name: Optional[str] = None, filter_duplicates: bool = True,
                           config_filename: str = "config.json", results_filename: str = "eval_results.json"):
    experiments_data = collect_data.collect_data_for_experiments(experiments_dir=experiments_dir, experiment_name=experiment_name,
                                                                 config_filename=config_filename, results_filename=results_filename)
    if filter_duplicates:
        experiments_data = collect_data.filter_duplicate_experiments(experiments_data, key_columns=key_columns,
                                                                     aggregate_columns=aggregate_columns)

    experiments_data_grouped = experiments_data.groupby(key_columns, dropna=False)
    data_by_experiment = {normalize_experiment_key(key): group for key, group in experiments_data_grouped}
    return data_by_experiment
