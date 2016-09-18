from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import lipnet_input
from confusion_matrix import ConfusionMatrix
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import helpers
from keras.models import load_model
import os


class ModelBasic(object):
    def __init__(self, verbose, compile_on_build=True):
        self.model = Sequential()
        self.verbose = verbose
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._scaler = StandardScaler()
        self._compile_on_build = compile_on_build

    def fit(self, train_set, test_set, nb_epoch):
        # features
        self.x_train = train_set.x
        self.x_train = self.preprocess_x(self.x_train)
        if len(self.x_train.shape) == 2:
            input_dim = self.x_train.shape[1]
        else:
            input_dim = self.x_train.shape[1:]
        output_dim = train_set.num_classes

        # test data
        if test_set is not None:
            self.x_test = test_set.x
            self.x_test = self.preprocess_x(self.x_test)
            self.y_test = self.get_y_for_train(test_set)

        # build model
        self.build_model(input_dim, output_dim)

        # target labels
        self.y_train = self.get_y_for_train(train_set)

    def train(self, train_set, nb_epoch):
        self.fit(train_set, nb_epoch)
        y_ = self.model.predict_proba(self.x_train, verbose=0)
        cf = ConfusionMatrix(y_, self.y_train)
        return cf

    def evaluate(self, test_set):
        x = test_set.x
        x = self.preprocess_x(x)
        y = test_set.y
        y = np_utils.to_categorical(y, test_set.num_classes)
        y_pred = self.model.predict_proba(x, verbose=0)
        cf = ConfusionMatrix(y_pred, y)
        return cf

    def get_y_for_train(self, train_set):
        y = train_set.y
        y = np_utils.to_categorical(y, train_set.num_classes)
        y = self.smooth_labels(y, 0.1)
        return y

    @staticmethod
    def smooth_labels(labels, eps):
        n_classes = labels.shape[1]
        return labels * (1 - eps) + (1 - labels) * eps / (n_classes - 1.0)

    def build_model(self, input_dim, output_dim):
        pass

    def preprocess_x(self, x):
        return x

    @property
    def description(self):
        return 'This is a base class for Keras classifiers'

    def save(self, path):
        savedir = os.path.basename(path)
        helpers.prepare_dir(savedir, empty=False)
        self.model.save(path)

    def restore(self, path):
        self.model = load_model(path)
