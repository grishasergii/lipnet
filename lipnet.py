import sys
import getopt
from lipnet_dataset import DatasetPDAugmented
import csv
from kfold import KFold
import numpy as np


def main():
    images = np.empty([10, 28, 28, 1], dtype=float)
    for i in range(10):
        images[i, :, :, :] = i
    images = images[:-1, :, :, :]
    pass

if __name__ == '__main__':
    main()