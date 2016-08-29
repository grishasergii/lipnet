from __future__ import division
import numpy as np


class Noise(object):

    def __init__(self, width, height, low=0, high=1):
        """

        :param width: int
        :param height: int
        :param low: float, optional
        :param high: float, optional
        """
        self._width = width
        self._height = height
        self._noise = np.random.uniform(low, high, (width, height))
        pass

    def _wrap_around(self, x, y):
        """
        Wraps coordinates around
        :param x: int or float
        :param y: int or float
        :return: two ints
        """
        x1 = (int(x) + self._width) % self._width
        y1 = (int(y) + self._height) % self._height
        return x1, y1

    def get_uniform_noise(self, x, y):
        """
        Returns uniform noise
        :param x: int
        :param y: int
        :return: float
        """
        x1, y1 = self._wrap_around(x, y)
        return self._noise[x1, y1]

    def get_smooth_noise(self, x, y, size):
        """
        Generates smooth noise using interpolation
        :param x: float
        :param y: float
        :param size: int > 1, smoothinf factor
        :return: float
        """
        if size == 0:
            return 0

        x = x / size
        y = y / size
        fract_x = x - int(x)
        fract_y = y - int(y)

        # do not do wrap around for performance reasons. x and y are always less or equal width and length
        # wrap around
        #x1, y1 = self._wrap_around(x, y)
        x1 = int(x)
        y1 = int(y)

        x2 = x1 - 1
        if x1 == 0:
            x2 = self._width - 1

        y2 = y1 - 1
        if y1 == 0:
            y2 = self._height - 1

        value = 0.0
        value += fract_x * fract_y * self._noise[y1, x1]
        value += (1 - fract_x) * fract_y * self._noise[y1, x2]
        value += fract_x * (1 - fract_y) * self._noise[y2, x1]
        value += (1 - fract_x) * (1 - fract_y) * self._noise[y2, x2]

        return value

    def turbulence(self, x, y, size):
        """
        Turbulence noise
        :param x: int or float
        :param y: int or float
        :param size: int, size of turbulence
        :return: float
        """
        if size == 0:
            return 0

        size_sum = 0
        value = 0
        while size >= 1:
            value += self.get_smooth_noise(x, y, size) * size
            size_sum += size
            size /= 2

        return value / size_sum
