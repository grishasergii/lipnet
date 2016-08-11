from sklearn.neighbors import NearestNeighbors
import numpy as np
from random import randint, uniform
import matplotlib.pyplot as plt


def smote(examples, n_synthetic_examples, n_neighbours=5):
    """
    Create synthetic examples using SMOTE method
    :param examples: 2d numpy array, each row represents one example
    :param n_synthetic_examples: int, number of synthetic examples to generate
    :param n_neighbours: int, number of nearest neighbours
    :return:
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbours, n_jobs=-1)
    neigh.fit(examples)
    nearest_neighbors = neigh.kneighbors(examples, return_distance=False)
    n_examples = examples.shape[0]

    examples_smote = np.zeros((n_synthetic_examples, examples.shape[1]))

    for i in xrange(n_synthetic_examples):
        # select random example index
        i_example = randint(0, n_examples-1)
        # select random nearest neighbor index
        i_neighbour = randint(0, n_neighbours-1)
        # get selected nearest neighbor of picked example
        neighbour = examples[nearest_neighbors[i_example, i_neighbour]]
        # compute difference between nearest neighbour and example
        diff = np.subtract(neighbour, examples[i_example])
        # select random gap
        gap = uniform(0, 1)
        # generate a synthetic example by apply difference with gap
        examples_smote[i] = np.add(examples[i_example], diff * gap)

    return examples_smote



