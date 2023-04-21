import numpy as np


def get_random_sample_from_dataset(dataset, size: int, seed: int):
    random_gen = np.random.default_rng(seed=seed)
    indices = random_gen.integers(low=0, high=len(dataset), size=size)
    return dataset.select(indices)
