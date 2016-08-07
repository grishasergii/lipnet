from __future__ import division
from abc import ABCMeta, abstractmethod
import pandas as pd
from random import shuffle
import os
from skimage import io, img_as_float
from skimage.transform import resize
import numpy as np
import sys
import math
import confusion_matrix as cf


class Batch:
    def __init__(self, data, labels, ids):
        """

        :param data:
        :param labels:
        :param ids:
        """
        assert data.shape[0] == labels.shape[0], "Number of data and corresponding labels must be the same"
        assert data.shape[0] == ids.shape[0], "Number of data and corresponding ids must be the same"
        self.images = data
        self.labels = labels
        self.ids = ids

    @property
    def size(self):
        return self.ids.shape[0]


class DatasetAbstract(object):
    """
    This is an abstract class that describes a dataset object.
    Any implementation of lipnet dataset must follow contract described here
    and implement all stated below methods.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def next_batch(self):
        """
        Yields a batch of data
        :return: batch
        """
        pass

    @abstractmethod
    def get_num_classes(self):
        """

        :return: integer, number of classes
        """
        pass

    @abstractmethod
    def get_count(self):
        """

        :return: integer, total number of examples in the dataset
        """
        pass


class DatasetPD(DatasetAbstract):
    """
    This is an implementation of lipnet dataset based on pandas dataframe and JSON
    """

    def __init__(self, path_to_json, path_to_img, batch_size=100, num_epochs=None, image_width=28, image_height=28):
        """

        :param path_to_json: full path to JSON file to be read into pandas dataframe
        :param path_to_img: full path to folder with images
        :param batch_size: integer, size of batch
        :param num_epochs: integer, number of times whole dataset can be processed, leave NOne for unlimited
        """
        super(DatasetPD, self).__init__()
        self._df = pd.read_json(path_to_json)
        self._shape = self._df.shape
        self._class_columns = [col for col in list(self._df) if col.startswith('Label')]
        self.num_epochs = num_epochs
        self._epoch_count = 0
        self._image_height = image_height
        self._image_width = image_width
        self._path_to_img = path_to_img
        self._batch_size = batch_size
        self._create_chunks()

        self.__prediction_columns = [c + '_prediction' for c in self._class_columns]
        for col in self.__prediction_columns:
            self._df[col] = 0

    @property
    def num_steps(self):
        if self.num_epochs is None:
            return sys.maxint
        return self.num_epochs * math.ceil(self.get_count() / self._batch_size)

    def chunks(self, items, chunk_size):
        """
        Creates a generator
        :param items: iterable, list of items to be transformed into chunks
        :param chunk_size: integer, number of items in a chunk. This is a maximum number of items, chunk can be smaller if
                    total length of l is not divisible by n
        :return: generator
        """
        for i in xrange(0, len(items), chunk_size):
            yield items[i:i + chunk_size]

    def next_batch(self):
        """
        Yields a batch of data
        :return: batch
        """
        try:
            ids = self.__chunks.next()
        except StopIteration:
            if self.__try_reset():
                return self.next_batch()
            else:
                return None
        return self._get_batch(ids)

    def _get_batch(self, ids):
        """
        Creates a batch from example ids
        :param ids: list of int, ids of examples
        :return: an instance of Batch class
        """
        img_names = self._df['Image'][self._df['Id'].isin(ids)]
        images = np.empty([len(img_names), self._image_width, self._image_height, 1], dtype=float)
        i = 0
        for f in img_names:
            filename = os.path.join(self._path_to_img, f)
            img = io.imread(filename)
            img = img_as_float(img)
            img = resize(img, (self._image_width, self._image_height))
            img = img.reshape((self._image_width, self._image_height, 1))
            images[i] = img
            i += 1
        labels = self._df[self._class_columns][self._df['Id'].isin(ids)].values
        return Batch(images, labels, np.array(ids))

    def __try_reset(self):
        """
        Resets chunks if epochs limit is not reached
        :return: boolean
        """
        if self.num_epochs is not None:
            if self._epoch_count >= (self.num_epochs - 1):
                return False

        self._epoch_count += 1
        self._create_chunks()
        return True

    def reset(self):
        """
        Resets epoch count and chunks generator
        :return: nothing
        """
        for col in self.__prediction_columns:
            self._df[col] = 0
        self._epoch_count = 0
        self._create_chunks()

    def print_stats(self):
        """
        Prints som dataframe statistics to console
        :return: nothing
        """
        print '{} columns and {} rows'.format(self._shape[1], self._shape[0])
        print self._df['Class'].value_counts()

    def _create_chunks(self):
        """
        Creates chunks
        :return: nothing, result is written to self.__chunks
        """
        list_of_ids = self._df['Id'].tolist()
        shuffle(list_of_ids)
        self.__chunks = self.chunks(list_of_ids, self._batch_size)

    def get_id_sorted_labels(self):
        """
        Returns labels sorted by example id
        :return: numpy array
        """
        return self._df.sort(columns=['Id'])[self._class_columns].values

    # DatasetAbstract methods

    def get_count(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return self._shape[0]

    def get_num_classes(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return len(self._class_columns)

    def set_predictions(self, ids, predictions):
        """
        Stores predictions in datatframe
        :param ids: list of ints representing ids
        :param predictions: 2d numpy array, number of columns must be equal to number of classes,
                            number of rows must be equal to length of ids
        :return: nothing
        """
        shape = predictions.shape
        assert len(shape) == 2, "Predictions must be a 2d array"
        assert shape[1] == self.get_num_classes(), "Number of classes in daatset and in predictions must be the same"
        assert ids.ndim == 1, "ids must be a vector"
        assert shape[0] == len(ids), "Number of ids and predictions must be the same"
        self._df.loc[self._df.Id.isin(ids), self.__prediction_columns] = predictions

    def evaluate(self):
        confusion_matrix = cf.ConfusionMatrix(self._df[self.__prediction_columns].values,
                                              self._df[self._class_columns].values)
        confusion_matrix.print_to_console()


class DatasetPDFeatures(DatasetPD):

    def _get_batch(self, ids):
        """
        Creates a batch from example ids
        :param ids: list of int, ids of examples
        :return: an instance of Batch class
        """
        feature_names = ['Area',
                         'Circularity',
                         'DiametersInPixels',
                         'Scale',
                         'Perimeter',
                         'Length',
                         'MembraneThickness'
                         ]
        data = self._df[feature_names][self._df['Id'].isin(ids)].values
        moments = np.array([np.array(xi) for xi in self._df['Moments'][self._df['Id'].isin(ids)].values])
        data = np.concatenate((data, moments), axis=1)
        labels = self._df[self._class_columns][self._df['Id'].isin(ids)].values
        return Batch(data, labels, np.array(ids))
