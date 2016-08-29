from __future__ import print_function
from liposome import LiposomeUnilamellar, LiposomeMultilamellar, LiposomeUncertain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import errno
import os
import sys


def make_liposomes(lip_class, n, out_dir):
    """

    :param lip_class:
    :param n:
    :param out_dir:
    :return:
    """
    try:
        os.makedirs(out_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    for i in xrange(n):
        liposome = lip_class(128, 128, prob_deviation=0.5)
        liposome.make()
        img_name = '{}.png'.format(i + 1)
        img_path = os.path.join(out_dir, img_name)
        print('\rSaving {}'.format(img_path), end='')
        mpimg.imsave(img_path, liposome.data, cmap='Greys_r', vmin=0, vmax=1)


def demo(n=10, out_dir='./out'):
    """
    Makes synthetic images of liposomes for demonstration
    :param n: int, optional, number of synthetic images of easch class
    :return: nothing, images are saved to current folder
    """
    make_liposomes(LiposomeUnilamellar, n, os.path.join(out_dir, 'unilamellar'))
    make_liposomes(LiposomeMultilamellar, n, os.path.join(out_dir, 'multilamellar'))
    make_liposomes(LiposomeUncertain, n, os.path.join(out_dir, 'uncertain'))
    print('')

if __name__ == '__main__':
    try:
        if len(sys.argv[1:]) == 1:
            n = int(sys.argv[1])
    except:
        n = 10
    demo(n=n)
