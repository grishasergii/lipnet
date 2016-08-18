from __future__ import division
import tensorflow as tf
from abc import abstractmethod
import math


def prime_powers(n):
    """
    Compute the factors of a positive integer
    from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in xrange(1, int(math.sqrt(n)) +1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)

class LayerAbstract(object):

    @staticmethod
    def apply(x):
        """
        Apply layer on input x
        :param x: tensor
        :return: tensor
        """
        return x

    @staticmethod
    def variable_summaries(var, name):
        """
        Attach summaries to a Tensor
        :param var: tensor, variable which summaries are reported for
        :param name: string, tensorboard scope
        :return: nothing
        """
        with tf.name_scope('Summaries') as scope:
            tf.scalar_summary(name + 'max', tf.reduce_max(var))
            tf.scalar_summary(name + 'min', tf.reduce_min(var))
            tf.histogram_summary(name, var)


class LayerConv2d(LayerAbstract):

    @staticmethod
    def _get_grid_dim(x):
        """
        Transforms x into product of two integers
        :param x: int
        :return: two ints
        """
        factors = prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i-1]

        i = len(factors) // 2
        return factors[i], factors[i]

    @staticmethod
    def _put_filters_on_grid(filters, (grid_x, grid_y), pad=1):
        """
        Transform conv weights to an image for visualizing purpose.
        based on https://gist.github.com/kukuruza/03731dc494603ceab0c5
        :param filters: tensor, conv weights
        :param (grid_x, grid_y): shape of the grid
        :param pad: int, padding between features
        :return: tensor
        """
        channels = 1
        # scale to [0, 1]
        x_min = tf.reduce_min(filters)
        x_max = tf.reduce_max(filters)
        _filters = (filters - x_min) / (x_max - x_min)

        # pad X and Y
        out = tf.pad(_filters, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]))

        y = filters.get_shape()[0] + pad*2
        x = filters.get_shape()[1] + pad*2

        out = tf.transpose(out, (3, 0, 1, 2))
        out = tf.reshape(out, tf.pack([grid_x, y * grid_y, x, channels]))
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, tf.pack([1, x * grid_x, y * grid_y, channels]))
        out = tf.transpose(out, (2, 1, 3, 0))
        out = tf.transpose(out, (3, 0, 1, 2))

        return out

    @classmethod
    def apply(cls, name, x, filter_shape, filter_num, stride):
        """
        Apply 2d convolution on tensor x
        :param name: string, layer name
        :param x: tensor
        :param stride: int
        :return: tensor
        """
        channels = x.get_shape()[3].value
        weights = tf.Variable(tf.random_normal(filter_shape + [channels, filter_num]))
        # visualize only first convolutional layer
        if channels == 1:
            weights_visualized = cls._put_filters_on_grid(weights,
                                                      cls._get_grid_dim(filter_num * channels))
            with tf.name_scope(name):
                with tf.name_scope('Features') as scope:
                    tf.image_summary(scope, weights_visualized, max_images=20)
        biases = tf.Variable(tf.random_normal([filter_num]))
        conv2d = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv2d = tf.nn.bias_add(conv2d, biases)
        return tf.nn.relu(conv2d)


class LayerFullyConnected(LayerAbstract):

    @staticmethod
    def _prepare_input_tensor(x):
        """
        Prepare input tensor for being processed in fullt connected layer
        :param x: tensor
        :return: tensor
        """
        shape = x.get_shape()
        if shape.ndims <= 2:
            # no need to reshape
            return x

        # reshape tensor x to make it 2d, 1 row per example
        n = 1
        for s in shape[1:]:
            n *= s.value
        out = tf.reshape(x, tf.pack([-1, n]))
        out.set_shape([None, n])
        return out

    @staticmethod
    def _get_output(layer_scope, x, w, b, activation_function=None, keep_prob=None):
        """
        combine input with weights and biases, apply activation function and dropout
        :param layer_scope: string, layer scope
        :param x: tensor, input to the layer
        :param w: tensor, weights
        :param b: tensor, biases
        :param activation_function: ref to activation function
        :param keep_prob: float, probability of keeping connection
        :return: tensor
        """
        with tf.name_scope(layer_scope):
            with tf.name_scope('Pre_activations') as scope:
                preactivate = tf.add(tf.matmul(x, w), b)
                LayerAbstract.variable_summaries(preactivate, scope)

            with tf.name_scope('Activations') as scope:
                out = preactivate
                if activation_function is not None:
                    out = activation_function(out)
                if keep_prob is not None:
                    out = tf.nn.dropout(out, keep_prob)
                LayerAbstract.variable_summaries(out, scope)

        return out

    @classmethod
    def apply(cls, name, x, nodes, activation_function, keep_prob=1):
        """
        Fully connected layer
        :param name: string, layer name for tensorboard
        :param x: tensor, input to the layer
        :param nodes: int, number of output nodes
        :param activation_function: ref to activation function
        :param keep_prob: float, probability of keeping connection
        :return: tensor
        """
        x = cls._prepare_input_tensor(x)
        n = x.get_shape()[1].value
        with tf.name_scope(name) as layer_scope:
            with tf.name_scope('Weights') as scope:
                weights = tf.Variable(tf.random_normal([n, nodes]))
                LayerAbstract.variable_summaries(weights, scope)
            with tf.name_scope('Biases') as scope:
                biases = tf.Variable(tf.random_normal([nodes]))
                LayerAbstract.variable_summaries(biases, scope)
            fc = cls._get_output(layer_scope, x, weights, biases, activation_function, keep_prob)
        return fc


class LayerOutput(LayerFullyConnected):

    @classmethod
    def apply(cls, name, x, nodes):
        """
        Last layer of the neural network, output layer
        :param name: string, layer name
        :param x: tensor, input
        :param nodes: int, number of nodes
        :return: tensor
        """
        with tf.name_scope(name) as layer_scope:
            fc = super(LayerOutput, cls).apply(layer_scope, x, nodes, None, keep_prob=None)
            max_logits = tf.reduce_max(tf.abs(fc), reduction_indices=[1])
            max_logits = tf.tile(max_logits, [nodes])
            max_logits = tf.reshape(max_logits, (nodes, -1))
            max_logits = tf.transpose(max_logits)
            with tf.name_scope('Scaled_activations') as scope:
                out = tf.div(fc, max_logits)
                LayerAbstract.variable_summaries(out, scope)
        return out


class LayerMaxPool(LayerAbstract):

    @staticmethod
    def apply(x, size, stride):
        """
        Perform max pooling
        :param x: tensor, input
        :param size: int, size of pooling
        :param stride: int, stride
        :return: tensor
        """
        return tf.nn.max_pool(x,
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')

