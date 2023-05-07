from typing import Optional
from typing import Union

import logging
import datetime
import os
import json

import torch
import transformers
import numpy as np

from dataproc import DatasetWrapper
from models import ModelWrapper

from self_learning import AugmentedDatasetWrapper
from self_learning import ModelWrapperForPseudoLabeling

import runexp.training_helpers as train_help

from runexp import ExperimentConfig


def save_eval_results(eval_results: dict, file_path: str):
    with open(file_path, "w+") as fd:
        json.dump(eval_results, fd)


def prepare_experiment_context(config: ExperimentConfig, dataset: Optional[DatasetWrapper] = None):
    logging.basicConfig(format='[%(levelname)s:%(process)d] %(asctime)s - %(message)s', level=logging.INFO)

    transformers.set_seed(config.random_seed)
    np.random.seed(config.random_seed)

    if config.output_dir is None:
        output_dir = "model_output/{}_{}".format(repr(config), datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        os.makedirs(output_dir)
        config.output_dir = output_dir
        logging.info("Will save results to directory %s", output_dir)

    logging.info("Preparing model")
    if config.self_learning_params.use_pseudo_labeling:
        model_class = ModelWrapperForPseudoLabeling
    else:
        model_class = ModelWrapper
    model = model_class.construct_with_config(config)

    if not dataset:
        logging.info("Preparing dataset")
        if config.self_learning_params.augmentation_type:
            dataset_class = AugmentedDatasetWrapper
        else:
            dataset_class = DatasetWrapper
        dataset: Union[DatasetWrapper, AugmentedDatasetWrapper] = dataset_class.construct_with_config(config)

        if config.labeled_train_set_size:
            dataset.split_train_data_into_labeled_and_unlabeled(
                labeled_dataset_size=config.labeled_train_set_size,
                unlabeled_dataset_size=config.self_learning_params.unlabeled_dataset_size,
                seed=config.random_seed)

        if config.self_learning_params.augmentation_type:
            logging.info("Augmenting dataset")
            dataset.augment(augmentation_type=config.self_learning_params.augmentation_type,
                            augmenter_kwargs=config.self_learning_params.augmenter_kwargs,
                            augmentation_scale=config.self_learning_params.augmentation_scale,
                            random_seed=config.random_seed,
                            device=config.cuda_device)

        dataset.preprocess_for_model(model)
    else:
        logging.info("Dataset already provided in function args")

    if config.max_target_length is None:
        config.max_target_length = dataset.max_target_length
        logging.info("Max target length is not set in config, using value inferred from dataset: %s", config.max_target_length)

    logging.info("Preparing training params")
    training_args = train_help.get_training_args(config=config)

    return model, dataset, training_args


def run_single_experiment(config: ExperimentConfig, dataset: Optional[DatasetWrapper] = None):
    model, dataset, training_args = prepare_experiment_context(config=config, dataset=dataset)

    if config.validation_set_size:
        validation_set = dataset.get_random_validation_data_subset(size=config.validation_set_size, seed=config.random_seed)
    else:
        validation_set = dataset.validation_data

    compute_metrics_fn = train_help.get_compute_metrics_fn(config=config, model=model, compute_additional_metrics=False)
    compute_metrics_fn_final = train_help.get_compute_metrics_fn(config=config, model=model, compute_additional_metrics=True)

    logging.info("Saving config to output dir")
    config.dump_to_file(file_path=os.path.join(config.output_dir, "config.json"))

    logging.info("Starting train_and_eval")
    eval_results = model.train_and_eval(train_data=dataset.preprocessed_train_data,
                                        unlabeled_data=dataset.preprocessed_unlabeled_train_data,
                                        validation_data=validation_set,
                                        training_arguments=training_args,
                                        compute_metrics_fn=compute_metrics_fn,
                                        compute_metrics_fn_final=compute_metrics_fn_final,
                                        config=config)

    logging.info("Saving results to output dir")
    save_eval_results(eval_results=eval_results, file_path=os.path.join(config.output_dir, "eval_results.json"))

    if config.delete_checkpoints:
        logging.info("Cleaning up checkpoints")
        train_help.delete_checkpoints(config=config)

    if config.do_use_gpu:
        torch.cuda.empty_cache()

    return model, eval_results
