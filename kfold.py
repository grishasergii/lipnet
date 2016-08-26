from __future__ import division
import pandas as pd
import math


class KFold:

    def __init__(self, k, path_to_json, path_to_img):
        """
        Stratified k-fold cross validation
        :param k: int, number of folds
        :param path_to_json: string
        :param path_to_img: string
        """
        self.k = k
        self._df = pd.read_json(path_to_json)
        self._df = pd.concat([self._df, pd.get_dummies(self._df['Class'], prefix='Label')], axis=1)
        self._df['Image'] = path_to_img + self._df['Image'].astype(str)

        self._df['Fold'] = 0

        labels = self._df['Class'].unique()
        self.len = self._df.shape[0]

        for label in labels:
            class_index = self._df[self._df['Class'] == label].index
            class_len = len(class_index)
            fold_size = int(math.ceil(class_len / k))
            fold = 0
            for i in xrange(0, class_len, fold_size):
                self._df.set_value(class_index[i:i + fold_size], 'Fold', fold)
                fold += 1

    def get_datasets(self, k):
        """
        Get train and test data sets, where one fold is used for evaluation and all other for training
        :param k: int, index of fold used for evaluation, starting from 0
        :return: two Pandas dataframes descendants, train set and test set
        """
        assert k >= 0, 'Fold index k = {} must be positive'.format(k)
        assert k < self.k, 'Fold index k = {} can not be greater than number of folds {}'.format(k, self.k)

        where = self._df.Fold == k
        test_set = self._df[where].copy()
        train_set = self._df[~where].copy()

        return train_set, test_set
