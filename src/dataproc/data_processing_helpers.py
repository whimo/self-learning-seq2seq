from typing import Optional
import numpy as np


def get_random_sample_indices(length: int, size: int, seed: int):
    random_gen = np.random.default_rng(seed=seed)
    indices = random_gen.choice(length, size=size)
    return indices


def get_random_split_indices(length: int, first_part_size: int, seed: int, second_part_size: Optional[int] = None):
    random_gen = np.random.default_rng(seed=seed)
    indices = np.arange(length)
    random_gen.shuffle(indices)

    if second_part_size is None:
        second_part_size = length - first_part_size
    return indices[:first_part_size], indices[first_part_size:first_part_size + second_part_size]


def get_random_sample_from_dataset(dataset, size: int, seed: int):
    indices = get_random_sample_indices(length=len(dataset), size=size, seed=seed)
    return dataset.select(indices)


def split_dataset_into_two_parts_randomly(dataset, first_part_size: int, seed: int, second_part_size: Optional[int] = None):
    first_part_indices, second_part_indices = get_random_split_indices(length=len(dataset),
                                                                       first_part_size=first_part_size,
                                                                       second_part_size=second_part_size,
                                                                       seed=seed)
    return dataset.select(first_part_indices), dataset.select(second_part_indices)
