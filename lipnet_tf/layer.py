from __future__ import division
import tensorflow as tf
from abc import abstractmethod
import math


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

    @staticmethod
    def weight_decay(var, wd):
        """
        Add weight decay operation to a Tensor
        :param var: Tensor
        :param wd: float
        :return: nothing, weight decay op is added to 'weight_losses' collection
        """
        pass
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('weight_losses', weight_decay)


class LayerConv2d(LayerAbstract):

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
        with tf.name_scope(name):
            with tf.name_scope('Weights') as scope:
                weights = tf.Variable(tf.random_normal(shape=filter_shape + [channels, filter_num], stddev=0.05),
                                      name='{}_weights'.format(name))
                tf.add_to_collection('conv_weights', weights)
                #cls.weight_decay(weights, 0.004)
                cls.variable_summaries(weights, scope)
            with tf.name_scope('Biases') as scope:
                biases = tf.Variable(tf.random_normal(shape=[filter_num], stddev=0.05))
                cls.variable_summaries(biases, scope)
            with tf.name_scope('Preactivations') as scope:
                preactivations = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
                preactivations = tf.nn.bias_add(preactivations, biases)
                cls.variable_summaries(preactivations, scope)
            with tf.name_scope('Activations') as scope:
                activations = tf.nn.relu(preactivations)
                cls.variable_summaries(activations, scope)
                tf.add_to_collection('conv_output', activations)
        return activations


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
            with tf.name_scope('Preactivations') as scope:
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
                weights = tf.Variable(tf.random_normal([n, nodes], stddev=0.05))
                #cls.weight_decay(weights, 0.004)
                LayerAbstract.variable_summaries(weights, scope)
            with tf.name_scope('Biases') as scope:
                biases = tf.Variable(tf.random_normal([nodes], stddev=0.05))
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
            out = fc
            #max_logits = tf.reduce_max(tf.abs(fc), reduction_indices=[1])
            #max_logits = tf.tile(max_logits, [nodes])
            #max_logits = tf.reshape(max_logits, (nodes, -1))
            #max_logits = tf.transpose(max_logits)
            with tf.name_scope('Scaled_activations') as scope:
                #out = tf.div(fc, max_logits)
                LayerAbstract.variable_summaries(out, scope)
        return out


class LayerMaxPool(LayerAbstract):

    @classmethod
    def apply(cls, name, x, size, stride):
        """
        Perform max pooling
        :param name: string, layer name
        :param x: tensor, input
        :param size: int, size of pooling
        :param stride: int, stride
        :return: tensor
        """
        with tf.name_scope(name) as scope:
            result = tf.nn.max_pool(x,
                                  ksize=[1, size, size, 1],
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')
            cls.variable_summaries(result, scope)
        return result

