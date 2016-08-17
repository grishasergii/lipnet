import tensorflow as tf
from abc import abstractmethod


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory
    :param name: name of the Variable
    :param initial_value: tensor
    :return: Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

    return var


def _activation_summary(x):
    """
    Helper to create summaries for an activation
    Creates a summary that provides a histogram of activations
    Creates a summary that measures the sparsity of activations
    :param x: tensor
    :return: nothing
    """
    tensor_name = x.op.name
    #tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.scalar_summary(tensor_name + '/max', tf.reduce_max(x))
    tf.scalar_summary(tensor_name + '/min', tf.reduce_min(x))


def fc_nn_layer(name, input_tensor, nodes, act, return_preactivations):
    """
    Reusable code for making a simple fully connected neural net layer.
    It does a matrix multiply, bias add, and then uses act function (sigmoid by default).
    It also sets up name scoping so that the resultant graph is easy to read and adds a number
    of summary operations.
    :param name: string, layer name, used for summaries and improves readability of tensorboard
    :param input_tensor: tensor, input to the layer
    :param nodes: int, number of nodes in the layer, i.e. number of outputs
    :param act: tf.nn activation function, relu by default
    :param return_preactivations: boolean, whether or not return preactivations
    :return: tensor, activations
    """
    n_inputs = input_tensor.get_shape()[1].value
    with tf.variable_scope(name) as scope:
        # create weights
        weights = tf.Variable(tf.random_normal([n_inputs, nodes], name='weights'))
        #weights = _variable_on_cpu('weights', [n_inputs, nodes], tf.random_normal_initializer)

        # create biases
        biases = tf.Variable(tf.random_normal([nodes], name='biases'))
        #biases = _variable_on_cpu('biases', [nodes], tf.random_normal_initializer())

        # multiply inputs with weights andd add bias
        preactivations = tf.add(tf.matmul(input_tensor, weights), biases)

        # apply activation function
        activations = act(preactivations, name=scope.name)

        # create summary op
        _activation_summary(activations)

        if return_preactivations:
            return preactivations, activations

        return activations


def conv_layer(name, input_tensor, filter_size, filter_num, stride=[1, 1], act=tf.nn.relu):
    """
    Reusable code for making a convolutional neural net layer.
    Good summary of what convolutional layer is
    http://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
    :param name: string, layer name, used for summaries and improves readability of tensorboard
    :param input_tensor: tensor, input to the layer
    :param filter_size: list of ints with length = 2, height and width of filter
    :param filter_num: int, number of output channels, or number of filters
    :param stride: list of ints with length = 2, steps in each direction. [1, 1] by default, i.e. whole image is used
    :param act: tf.nn activation function, relu by default
    :return: tensor, activations
    """
    # get number of channels in input
    input_channels = input_tensor.get_shape()[3].value
    with tf.variable_scope(name) as scope:
        # prepare dimension of filter
        filter_dim = filter_size + [input_channels, filter_num]

        # make filter variable
        filter = tf.Variable(tf.random_normal(filter_dim, stddev=0), name='filter')

        # add 1 to the right and to the left of stride
        # first 1 corresponds to batch, images
        # last 1 corresponds to channels
        # we generally do not want to skip any images iin the batch or any channels
        strides = [1] + stride + [1]

        # make convolution operation
        conv = tf.nn.conv2d(input_tensor, filter, strides, padding='SAME')

        # make and apply biases
        biases = tf.Variable(tf.random_normal([filter_num], name='biases'))
        #biases = _variable_on_cpu('biases', [filter_num], tf.random_normal_initializer())
        preactivations = tf.nn.bias_add(conv, biases)

        # calculate activations
        activations = act(preactivations, name=scope.name)

        # create summary op
        _activation_summary(activations)

        return activations


class LayerAbstract(object):

    @staticmethod
    def apply(x):
        """
        Apply layer on input x
        :param x: tensor
        :return: tensor
        """
        return x


class LayerConv2d(LayerAbstract):

    @staticmethod
    def apply(x, filter_shape, filter_num, stride):
        """
        Apply 2d convolution on tensor x
        :param x: tensor
        :param stride: int
        :return: tensor
        """
        channels = x.get_shape()[3].value
        weights = tf.Variable(tf.random_normal(filter_shape + [channels, filter_num]))
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
    def _get_output(x, w, b, activation_function=None, keep_prob=None):
        """
        combine input with weights and biases, apply activation function and dropout
        :param x: tensor, input to the layer
        :param w: tensor, weights
        :param b: tensor, biases
        :param activation_function: ref to activation function
        :param keep_prob: float, probability of keeping connection
        :return: tensor
        """
        out = tf.add(tf.matmul(x, w), b)

        if activation_function is not None:
            out = activation_function(out)

        if keep_prob is not None:
            out = tf.nn.dropout(out, keep_prob)

        return out

    @classmethod
    def apply(cls, x, nodes, activation_function, keep_prob=1):
        """
        Fully connected layer
        :param x: tensor, input to the layer
        :param nodes: int, number of output nodes
        :param activation_function: ref to activation function
        :param keep_prob: float, probability of keeping connection
        :return: tensor
        """
        x = cls._prepare_input_tensor(x)
        n = x.get_shape()[1].value
        weights = tf.Variable(tf.random_normal([n, nodes]))
        biases = tf.Variable(tf.random_normal([nodes]))
        fc = cls._get_output(x, weights, biases, activation_function, keep_prob)
        return fc


class LayerOutput(LayerFullyConnected):

    @classmethod
    def apply(cls, x, nodes):
        """
        Last layer of the neural network, output layer
        :param x: tensor, input
        :param nodes: int, number of nodes
        :return: tensor
        """
        fc = super(LayerOutput, cls).apply(x, nodes, None, keep_prob=None, name=name)
        max_logits = tf.reduce_max(tf.abs(fc), reduction_indices=[1])
        max_logits = tf.tile(max_logits, [nodes])
        max_logits = tf.reshape(max_logits, (nodes, -1))
        max_logits = tf.transpose(max_logits)
        fc = tf.div(fc, max_logits)

        return fc


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

