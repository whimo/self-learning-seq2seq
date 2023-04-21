import logging
import datetime
import os
import json

import torch
import transformers

from dataproc import DatasetWrapper
from models import ModelWrapper

import runexp.training_helpers as train_help

from runexp import ExperimentConfig


def save_eval_results(eval_results: dict, file_path: str):
    with open(file_path, "w+") as fd:
        json.dump(eval_results, fd)


def run_single_experiment(config: ExperimentConfig):
    logging.basicConfig(format='[%(levelname)s:%(process)d] %(asctime)s - %(message)s', level=logging.INFO)

    transformers.set_seed(config.random_seed)

    if config.output_dir is None:
        output_dir = "model_output/{}_{}".format(repr(config), datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        os.makedirs(output_dir)
        config.output_dir = output_dir
        logging.info("Will save results to directory %s", output_dir)

    logging.info("Preparing model")
    model = ModelWrapper.construct_with_config(config)
    logging.info("Preparing dataset")
    dataset = DatasetWrapper.construct_with_config(config)
    dataset.preprocess_for_model(model, max_input_length=config.max_input_length, max_target_length=config.max_target_length)

    logging.info("Preparing training params")
    training_args = train_help.get_training_args(config)
    if config.labeled_train_set_size:
        training_subset = dataset.get_random_train_data_subset(size=config.labeled_train_set_size, seed=config.random_seed)
    else:
        training_subset = dataset.train_data

    if config.validation_set_size:
        validation_subset = dataset.get_random_validation_data_subset(size=config.validation_set_size, seed=config.random_seed)
    else:
        validation_subset = dataset.validation_data

    compute_metrics_fn = train_help.get_compute_metrics_fn(config=config, model=model)

    logging.info("Saving config to output dir")
    config.dump_to_file(file_path=os.path.join(config.output_dir, "config.json"))

    logging.info("Starting train_and_eval")
    eval_results = model.train_and_eval(train_data=training_subset,
                                        validation_data=validation_subset,
                                        training_arguments=training_args,
                                        compute_metrics_fn=compute_metrics_fn)

    logging.info("Saving results to output dir")
    save_eval_results(eval_results=eval_results, file_path=os.path.join(config.output_dir, "eval_results.json"))

    if config.do_use_gpu:
        torch.cuda.empty_cache()

    return model, eval_results
