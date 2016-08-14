from sklearn.neighbors import NearestNeighbors
import numpy as np
from random import randint, uniform
import matplotlib.pyplot as plt


def _combine_by_diff(example, neighbor):
    diff = np.subtract(neighbor, example)
    # select random gap
    gap = uniform(0, 1)
    # generate a synthetic example by apply difference with gap
    return np.add(example, diff * gap)

def _combine_by_blend(example, neighbor):
    alpha = uniform(0.25, 0.75)
    return example * (1.0 - alpha) + neighbor * alpha


def smote(examples, n_synthetic_examples, n_neighbours=5, return_parent_ids=False):
    """
    Create synthetic examples using SMOTE method
    :param examples: 2d numpy array, each row represents one example
    :param n_synthetic_examples: int, number of synthetic examples to generate
    :param n_neighbours: int, number of nearest neighbours
    :param return_parent_ids: boolean, set to true to get parents ids of synthetic examples
    :return:
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbours+1, n_jobs=-1)

    neigh.fit(examples)
    nearest_neighbors = neigh.kneighbors(examples, return_distance=False)

    n_examples = examples.shape[0]

    examples_smote = np.zeros((n_synthetic_examples, examples.shape[1]))
    if return_parent_ids:
        parent_ids = np.zeros((n_synthetic_examples, 2))

    for i in xrange(n_synthetic_examples):
        # select random example index
        example_idx = randint(0, n_examples-1)
        # select random nearest neighbor index
        i_neighbour = randint(1, n_neighbours)
        neighbour_idx = nearest_neighbors[example_idx, i_neighbour]

        # get selected nearest neighbor of picked example
        neighbour = examples[neighbour_idx]

        # generate a synthetic example by some of combining methods
        s = _combine_by_diff(examples[example_idx], neighbour)
        examples_smote[i] = s

        if return_parent_ids:
            parent_ids[i, :] = [example_idx, neighbour_idx]

    if return_parent_ids:
        return examples_smote, parent_ids

    return examples_smote



