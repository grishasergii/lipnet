from noise import Noise
import numpy as np
from abc import abstractmethod
import math
import random


class LiposomeBasic(object):

    def __init__(self, width, height, prob_deviation=0):
        """

        :param width: int
        :param height: int
        :param prob_deviation: float, optional, probability of some deviation
        """
        self.width = width
        self.height = height
        self._noise = Noise(width, height, low=0.2, high=0.7)
        self.data = self._get_background()
        self._prob_deviation = prob_deviation
        self._prob_deviation_internal = 0.3

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

    #@abstractmethod
    def _deviate(self):
        if random.random() > 0.4:
            self._deviation_spot()

        if random.random() > 0.97:
            self._deviation_patch()

        if random.random() < self._prob_deviation_internal:
            self._deviation_overlay_internal(self._center_x, self._center_y, self._radius)

        if random.random() > 0.5:
            self._deviation_arc_unilameral()

        if random.random() > 0.9:
            self._deviation_arc_multilamellar()

    def _draw_circle(self, radius, thickness,
                     center_x=0, center_y=0, a=1, b=1,
                     turb_size=0, turb_power=0):
        """

        :param radius:
        :param thickness:
        :param center_x:
        :param center_y:
        :param a: float, optional, ellipse: radius on x axis
        :param b: float, optional, ellipse: radius on y axis
        :return:
        """
        noise = Noise(self.width, self.height)
        a = 1 / (a * a)
        b = 1 / (b * b)
        for x in xrange(self.width):
            for y in xrange(self.height):
                xv = (x - self.width / 2.0) / self.width
                yv = (y - self.height / 2.0) / self.height
                dist = math.sqrt((xv - center_x) * (xv - center_x) * a +
                                 (yv - center_y) * (yv - center_y) * b)
                dist += turb_power * noise.turbulence(x, y, turb_size)
                if ((radius - thickness) <= dist) and \
                        (dist <= (radius + thickness)):
                    theta = np.interp(dist,
                                      [radius - thickness, radius + thickness],
                                      [0, math.pi])
                    multiplier = random.uniform(0.2, 0.25)
                    value = math.sin(theta) * multiplier
                    self.data[x, y] -= value

    def make(self):
        self._draw()
        if random.random() < self._prob_deviation:
            self._deviate()

    def _deviation_spot(self):
        # add a spot like on image 544825
        p = random.random()
        n = 1
        if p > 0.5:
            n = 2
        if p > 0.75:
            n = 3
        for _ in xrange(n):
            x = random.uniform(-0.3, 0.3)
            y = random.uniform(-0.3, 0.3)
            radius = random.uniform(0.01, 0.03)
            self._draw_circle(radius, 0.05, center_x=x, center_y=y)

    def _deviation_patch(self):
        # a patch like on image 545302
        x = random.uniform(-0.3, 0.3)
        y = random.uniform(-0.3, 0.3)
        radius = 0.05
        thickness = 0.6
        turb_size = 4
        turb_power = 0.1
        self._draw_circle(radius, thickness, center_x=x, center_y=y, turb_size=turb_size, turb_power=turb_power)

    def _deviation_overlay_internal(self, center_x, center_y, center_radius, n=None, radius=None):
        """
        Overlay of liposomes like on image 539202
        :param n: int, optional, number of inscribed liposomes, randomized if None
        :return: nothing
        """
        if n is None:
            n = random.randint(1, 2)
        for _ in xrange(n):
            if radius is None:
                radius = random.uniform(0.2, 0.3)
            thickness = random.uniform(0.04, 0.05)
            theta = random.randint(1, 360)
            x = center_x + (center_radius - radius) * math.cos(theta)
            y = center_y + (center_radius - radius) * math.sin(theta)
            a = random.uniform(0.7, 1.3)
            b = random.uniform(0.7, 1.3)
            self._draw_circle(radius, thickness, center_y=y, center_x=x,
                              turb_size=32, turb_power=0.1, a=a, b=b)

    def _deviation_arc_unilameral(self, n=None):
        """
        Draw an arc(s) as on image 545216, 545413, 547997
        :param n: int, optional, number of arcs, randomized between 1 and 5 if None
        :return:
        """
        if n is None:
            p = random.random()
            n = 1
            if p > 0.4:
                n = 2
            if p > 0.6:
                n = 3
            if p > 0.9:
                n = 4
            if p > 0.97:
                n = 5
        thetas = np.linspace(0, 360, 6)
        thetas = random.sample(thetas, n)
        for theta in thetas:
            radius = random.uniform(0.4, 0.7)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            thickness = random.uniform(0.04, 0.05)
            a = random.uniform(0.8, 1.2)
            b = random.uniform(0.8, 1.2)
            radius = random.uniform(0.2, 0.4)
            self._draw_circle(radius, thickness, center_y=y, center_x=x, a=a, b=b)

    def _deviation_arc_multilamellar(self, n=None):
        """
        Deviation like on image 546292
        :param n: int, optional, number of arcs, randomized between 2 and 3 if None
        :return:
        """
        if n is None:
            n = 1
            if random.random() > 0.5:
                n = 2
        thetas = np.linspace(0, 360, num=6)
        thetas = random.sample(thetas, n)
        for theta in thetas:
            circles = 2
            if random.random() > 0.5:
                circles = 3

            radius = random.uniform(0.5, 0.7)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            thickness = random.uniform(0.03, 0.05)
            a = random.uniform(0.8, 1.2)
            b = random.uniform(0.8, 1.2)
            radius = random.uniform(0.4, 0.6)
            for _ in xrange(circles):
                self._draw_circle(radius, thickness, center_y=y, center_x=x, a=a, b=b)
                radius -= random.uniform(0.08, 0.11)


class LiposomeUnilamellar(LiposomeBasic):

    def _draw(self):
        thickness = random.uniform(0.05, 0.06)

        a = 1
        b = 1
        if random.random() <= self._prob_deviation:
            a = random.uniform(0.9, 1.1)
            b = random.uniform(0.9, 1.1)

        turb_size = 64
        turb_power = 0.1
        self._radius = random.uniform(0.47, 0.5)
        self._center_x = random.uniform(-0.1, 0.1)
        self._center_y = random.uniform(-0.1, 0.1)
        self._draw_circle(radius=self._radius, thickness=thickness, a=a, b=b,
                          turb_size=turb_size, turb_power=turb_power,
                          center_x=self._center_x, center_y=self._center_y)


class LiposomeMultilamellar(LiposomeUnilamellar):

    def __init__(self, width, height, num_circles=None, prob_deviation=0):
        """

        :param width:
        :param height:
        :param num_circles: int, optional, number of circles, 2 by default, set to None to randomize
        """
        super(LiposomeMultilamellar, self).__init__(width, height, prob_deviation=prob_deviation)
        self._prob_deviation_internal = 0.6

        self.num_circles = num_circles
        if self.num_circles is None:
            self.num_circles = 2
            if random.random() > 0.7:
                self.num_circles = 3
            if random.random() > 0.97:
                self.num_circles = 4

    def _draw(self):
        self._radius = random.uniform(0.47, 0.5)
        radius = self._radius
        self._center_x = random.uniform(-0.05, 0.05)
        self._center_y = random.uniform(-0.05, 0.05)
        for _ in xrange(self.num_circles):
            thickness = random.uniform(0.05, 0.06)
            turb_size = 64
            turb_power = 0.1
            self._draw_circle(radius=radius, thickness=thickness,
                              center_x=self._center_x, center_y=self._center_y,
                              turb_size=turb_size, turb_power=turb_power)
            radius -= random.uniform(0.14, 0.2)


class LiposomeUncertain(LiposomeBasic):

    def _deviate(self):
        pass

    def _draw(self):
        thickness = random.uniform(0.05, 0.06)

        a = 1
        b = 1
        if random.random() <= self._prob_deviation:
            a = random.uniform(0.9, 1.1)
            b = random.uniform(0.9, 1.1)

        turb_size = 64
        turb_power = 0.1
        self._radius = random.uniform(0.47, 0.5)
        self._center_x = random.uniform(-0.1, 0.1)
        self._center_y = random.uniform(-0.1, 0.1)
        self._draw_circle(radius=self._radius, thickness=thickness, a=a, b=b,
                          turb_size=turb_size, turb_power=turb_power,
                          center_x=self._center_x, center_y=self._center_y)

        self._deviation_overlay_internal(self._center_x, self._center_y, self._radius, n=1,
                                         radius=random.uniform(0.3, 0.4))
        if random.random() > 0.8:
            self._deviation_overlay_internal(self._center_x, self._center_y, self._radius, n=1,
                                             radius=random.uniform(0.3, 0.4))
        if random.random() > 0.6:
            self._draw_circle(radius=random.uniform(0.15, 0.25),
                              thickness=thickness,
                              center_x=random.uniform(-0.2, 0.2),
                              center_y=random.uniform(-0.2, 0.2))

        if random.random() > 0.7:
            self._deviation_arc_unilameral(n=3)


