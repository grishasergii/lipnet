from abc import ABCMeta, abstractmethod
import pandas as pd


class DatasetAbstract(object):
    """
    This is an abstract class that describes a dataset object.
    Any implementation of lipnet dataset must follow contract described here
    and implement all stated below methods.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_num_classes(self):
        """

        :return: integer, number of classes
        """
        pass

    @abstractmethod
    def get_image_names(self):
        """

        :return: numpy 1D array of strings, corresponding to image names
        """
        pass

    @abstractmethod
    def get_labels(self):
        """

        :return: numpy 2D array, one-hot encoded labels
        """
        pass

    @abstractmethod
    def get_examples_count(self):
        """

        :return: integer, total number of examples in the dataset
        """


class DatasetPD(DatasetAbstract):
    """
    This is an implementation of lipnet dataset based on pandas dataframe and JSON
    """
    __df = pd.DataFrame()
    __shape = None
    __class_columns = None

    def __init__(self, path_to_json):
        """

        :param path_to_json: full path to JSON file to be read into pandas dataframe
        """
        super(DatasetPD, self).__init__()
        self.__df = pd.read_json(path_to_json)
        self.__shape = self.__df.shape
        self.__class_columns = [col for col in list(self.__df) if col.startswith('Label')]

    def get_examples_count(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return self.__shape[0]

    def get_num_classes(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return len(self.__class_columns)

    def get_image_names(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return self.__df['Image'].values

    def get_labels(self):
        """
        See description in DatasetAbstract
        :return:
        """
        return self.__df[self.__class_columns].values

    def print_stats(self):
        """
        Prints som dataframe statistics to console
        :return: nothing
        """
        print '{} columns and {} rows'.format(self.__shape[1], self.__shape[0])
        print self.__df['Class'].value_counts()