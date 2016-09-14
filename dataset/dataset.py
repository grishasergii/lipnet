from __future__ import division
from abc import ABCMeta, abstractmethod
import pandas as pd
import confusion_matrix as cf
import numpy as np
from smote import smote
import math
from scipy import stats, signal


# move it to separate module!
class Batch(object):
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


class DatasetBasic(object):

    def __init__(self, df, batch_size=None):
        self._df = df.copy()

        # replace class names with integers
        self._df['Class'] = self._df['Class'].replace(to_replace=['Unilamellar', 'Multilamellar', 'Uncertain', 'Empty',
                                                                 'Full'],
                                                      value=[0, 1, 2, 0, 1])

        # prepare class columns
        self._class_columns = [col for col in list(self._df) if col.startswith('Label')]
        if len(self._class_columns) == 0:
            self._df = pd.concat([self._df, pd.get_dummies(self._df['Class'], prefix='Label')], axis=1)
            self._class_columns = [col for col in list(self._df) if col.startswith('Label')]

        # do label smoothing
        # as described in Deep Learning book section 7.5.1
        eps = 0.1
        ix = self._class_columns
        self._df[ix] = self._df[ix] * (1 - eps) + (1 - self._df[ix]) * eps / (len(ix) - 1)

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.count

        # make columns for predictions
        self._prediction_columns = [c + '_prediction' for c in self._class_columns]
        for col in self._prediction_columns:
            self._df[col] = 0

    @property
    def count(self):
        return self._df.shape[0]

    @property
    def batches_count(self):
        return int(math.ceil(self.count / self.batch_size))

    @property
    def num_classes(self):
        return len(self._class_columns)

    @property
    def confusion_matrix(self):
        return cf.ConfusionMatrix(self._df[self._prediction_columns].values,
                                  self._df[self._class_columns].values)

    @property
    def input_shape(self):
        return None

    @property
    def balanced_class_weights(self):
        n_samples = len(self._df)
        return float(n_samples) / (self.num_classes * np.bincount(self._df.Class.values))

    def iterate_minibatches(self, shuffle_=False):
        ids = self._df.Id.values.copy()
        if shuffle_:
            np.random.shuffle(ids)
        start_idx = 0
        while start_idx < ids.shape[0]:
            excerpt = ids[start_idx:start_idx + self.batch_size]
            minibatch = self._get_batch(excerpt)
            start_idx += self.batch_size
            yield minibatch

    @abstractmethod
    def _get_batch(self, ids):
        pass

    @property
    def x(self):
        return []

    @property
    def y(self):
        return self._df['Class'].values.copy()

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
        assert shape[1] == self.num_classes, "Number of classes in dataset and in predictions must be the same"
        assert ids.ndim == 1, "ids must be a vector"
        assert shape[0] == len(ids), "Number of ids and predictions must be the same"
        for i, _id in enumerate(ids):
            try:
                ix = self._df.loc[self._df.Id == _id].index
                self._df.loc[ix, self._prediction_columns] = predictions[i]
            except TypeError:
                pass

    @property
    def confusion_matrix(self):
        return cf.ConfusionMatrix(self._df[self._prediction_columns].values,
                                  self._df[self._class_columns].values)


class DatasetFeatures(DatasetBasic):

    def __init__(self, df, batch_size=None, do_oversampling=True):
        super(DatasetFeatures, self).__init__(df, batch_size=batch_size)

        self.do_oversampling = do_oversampling
        self.feature_names = ['Area',
                              'Circularity',
                              'Perimeter',
                              'Length',
                              'MaximumWidth',
                              'SignalToNoise',
                              'M20',
                              'M02',
                              'M30',
                              'M03'
                              ]
        # split moments column into separate columns
        self._df[['M20', 'M02', 'M30', 'M03']] = self._df['Moments'].apply(lambda x: pd.Series(x))
        # normalize features
        self._df[self.feature_names] = self._df[self.feature_names].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

        if do_oversampling:
            self.oversample()

    @classmethod
    def from_json(cls, path_to_json, batch_size=None, do_oversampling=True):
        df = pd.read_json(path_to_json)
        return cls(df, batch_size=batch_size, do_oversampling=do_oversampling)

    def _get_batch(self, ids):
        data = self._df[self.feature_names][self._df['Id'].isin(ids)].values
        labels = self._df[self._class_columns][self._df['Id'].isin(ids)].values
        return Batch(data, labels, np.array(ids))

    @property
    def input_shape(self):
        return [len(self.feature_names)]

    def oversample(self):
        # determine majority and minority classes
        class_counts = self._df['Class'].value_counts()
        majority_class = class_counts.index[0]
        minority_classes = class_counts.index[1:]

        new_id = self._df['Id'].max() + 1
        # generate synthetic examples
        for i, class_label in enumerate(minority_classes):
            n = class_counts[majority_class] - class_counts[class_label]
            class_targets = self._df.loc[self._df.Class == class_label][self._class_columns].values[0, :].copy()
            class_targets =np.tile(class_targets, [n, 1])
            synthetic_data = smote(self._df[self.feature_names].values.copy(), n)
            ids = np.arange(new_id, new_id + n)
            ids = np.reshape(ids, [n, 1])
            synthetic_data = np.concatenate((ids, synthetic_data, class_targets), axis=1)
            df = pd.DataFrame(synthetic_data, columns=['Id'] + self.feature_names + self._class_columns)
            self._df = self._df.append(df, ignore_index=True)
            new_id += n

    def _resample_rdp(self, n):
        """
        Resample Radial Density Profile
        :param n: int
        :return: nothing
        """
        columns = []
        for i in xrange(n):
            columns += ['edp_{}'.format(i)]
        self.feature_names += columns

        # edp = [None] * len(self._df)
        edp = np.zeros([len(self._df), n])
        for i, v in enumerate(self._df.RadialDensityProfile.values):
            x = np.zeros([len(v)])
            for j, v_ in enumerate(v):
                x[j] = v_[0]
            edp[i] = signal.resample(x, n)
        df = pd.DataFrame(data=edp, columns=columns, index=self._df.index)
        self._df = pd.concat([self._df, df], axis=1)

    def _resample_edp(self, n):
        """
        Resample Edge Density Profile
        :param n: int
        :return: nothing
        """
        columns = []
        for i in xrange(n):
            columns += ['edp_{}'.format(i)]
        self.feature_names += columns

        # edp = [None] * len(self._df)
        rdp = np.zeros([len(self._df), n])
        for i, v in enumerate(self._df.EdgeDensityProfile.values):
            x = np.zeros([len(v)])
            for j, v_ in enumerate(v):
                x[j] = v_[0]
            rdp[i] = signal.resample(x, n)
        df = pd.DataFrame(data=rdp, columns=columns, index=self._df.index)
        self._df = pd.concat([self._df, df], axis=1)

    def _transform_histogram(self):
        h_n = len(self._df.Histogram.values[0])
        columns = []
        for i in xrange(h_n):
            columns += ['histogram_{}'.format(i)]

        histogram = np.zeros([len(self._df), h_n])
        for i, v in enumerate(self._df.Histogram.values):
            histogram[i] = np.array(v)
        df = pd.DataFrame(columns=columns, data=histogram, index=self._df.index)
        self._df = pd.concat([self._df, df], axis=1)

        self.feature_names += columns


class DatasetEDP(DatasetFeatures):

    def __init__(self, df, batch_size=None, do_oversampling=True):
        super(DatasetFeatures, self).__init__(df, batch_size=batch_size)

        self.do_oversampling = do_oversampling
        self.feature_names = []
        self._n_sample = 69
        self._resample_edp(self._n_sample)


class DatasetRDP(DatasetFeatures):

    def __init__(self, df, batch_size=None, do_oversampling=True):
        super(DatasetFeatures, self).__init__(df, batch_size=batch_size)

        self.do_oversampling = do_oversampling
        self.feature_names = []
        self._n_sample = 69
        self._resample_rdp(self._n_sample)

    @property
    def input_shape(self):
        return [self._n_sample]


class DatasetVironovaSVM(DatasetFeatures):

    def __init__(self, df, batch_size=None, do_oversampling=True):
        super(DatasetVironovaSVM, self).__init__(df,
                                                 batch_size=batch_size,
                                                 do_oversampling=do_oversampling)

        self.do_oversampling = do_oversampling
        self.feature_names = ['Area',
                              'Circularity',
                              'M20',
                              'M02',
                              'M30',
                              'M03']
        self._n_sample = 80

        self._resample_edp(self._n_sample)
        #self._transform_histogram()

        pass

    @property
    def input_shape(self):
        return [len(self.feature_names)]

