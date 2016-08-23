from noise import Noise
import numpy as np
from abc import abstractmethod
import math



class LiposomeBasic(object):

    def __init__(self, width, height):
        """

        :param width: int
        :param height: int
        """
        self.width = width
        self.height = height
        self._noise = Noise(width, height, low=0.2, high=0.7)
        self.data = self._get_background()
        self._draw()

    def _get_background(self):
        """
        Makes background of microscope image
        :return: numpy 2d array of shape(self.width, self.height)
        """
        data = np.zeros((self.width, self.height))
        turbulence_size = 4
        for x in xrange(self.width):
            for y in xrange(self.height):
                data[x, y] = self._noise.turbulence(x, y, turbulence_size)
        return data

    @abstractmethod
    def _draw(self):
      pass


class LiposomeUnilamellar(LiposomeBasic):

    def _draw_circle(self, radius, thickness, center_x=0, center_y=0):
        """

        :param radius:
        :param thickness:
        :param center_x:
        :param center_y:
        :return:
        """
        noise = Noise(self.width, self.height)
        turb_size = 64
        turb_power = 0.1
        for x in xrange(self.width):
            for y in xrange(self.height):
                xv = (x - self.width / 2.0) / self.width
                yv = (y - self.height / 2.0) / self.height
                dist = math.sqrt(xv * xv + yv * yv)
                dist += turb_power * noise.turbulence(x, y, turb_size)
                if ((radius - thickness) <= dist) and \
                   (dist <= (radius + thickness)):
                    theta = np.interp(dist,
                                      [radius - thickness, radius + thickness],
                                      [0, math.pi])
                    value = math.sin(theta) / 4.5
                    self.data[x, y] -= value

    def _draw(self):
        self._draw_circle(radius=0.45, thickness=0.035)


class LiposomeMultilamellar(LiposomeUnilamellar):

    def _draw(self):
        self._draw_circle(radius=0.47, thickness=0.045)
        self._draw_circle(radius=0.35, thickness=0.036)
        self._draw_circle(radius=0.25, thickness=0.037)