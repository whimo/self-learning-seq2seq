from typing import Optional
import numpy as np


def get_random_sample_from_dataset(dataset, size: int, seed: int):
    random_gen = np.random.default_rng(seed=seed)
    indices = random_gen.choice(len(dataset), size=size)
    return dataset.select(indices)


def split_dataset_into_two_parts_randomly(dataset, first_part_size: int, seed: int, second_part_size: Optional[int] = None):
    random_gen = np.random.default_rng(seed=seed)
    indices = np.arange(len(dataset))
    random_gen.shuffle(indices)

    if second_part_size is None:
        second_part_size = len(dataset) - first_part_size
    return dataset.select(indices[:first_part_size]), dataset.select(indices[first_part_size:first_part_size + second_part_size])
