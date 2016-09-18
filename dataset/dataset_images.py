from __future__ import division
from dataset import DatasetBasic
import pandas as pd
from skimage import io, img_as_float
from skimage.transform import resize
import numpy as np
import math


class DatasetImages(DatasetBasic):

    def __init__(self, df, img_size):
        df['Image_data'] = None
        # read images to the dataframe
        #"""
        for idx, row in df.iterrows():
            img = self.read_image(row.Image, img_size)
            df.set_value(idx, 'Image_data', img)

        # delete all rows where image_data is None
        df = df[df.Image_data != None]
        #"""
        super(DatasetImages, self).__init__(df)

        self.img_size = img_size
        self.feature_names = ['Image_data']
        self._df = self._df[self.feature_names + self._class_columns + ['Class']]

    @classmethod
    def from_json(cls, path_to_json, path_to_img, img_size=None):
        """
        Read dataset from json
        :param path_to_json: string, path to kson file
        :param path_to_img: string, path to folder where images are located
        :param img_size: tuple of two ints, optional, new size of images
        :return: Dataset object
        """
        df = pd.read_json(path_to_json)

        # append path to images to column that contains image names
        df['Image'] = path_to_img + df['Image'].astype(str)

        return cls(df, img_size)

    @property
    def x(self):
        x_ = np.empty([len(self._df),
                       1,
                       self.img_size[0],
                       self.img_size[1]])
        for i, row in enumerate(self._df.iterrows()):
            x_[i, :] = row[1].Image_data
        return x_

    @staticmethod
    def read_image(image_name, img_size=None):
        """
        Read image from file
        :param image_name: string, image full name including path
        :param img_size: tuple of two ints, optional, specify if you want to resize image
        :return: numpy 2d array
        """
        filename = image_name
        try:
            img = io.imread(filename)
        except IOError:
            return None
        img = img_as_float(img)
        if len(img.shape) > 2:
            img = img[:, :, 0]

        if img_size is not None:
            img = resize(img, (img_size[0], img_size[1]))
            img = img.reshape((1, img_size[0], img_size[1]))

        return img
