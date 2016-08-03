import tensorflow as tf
from abc import abstractmethod
from tensorflow.python.ops import  control_flow_ops


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
        weights = _variable_on_cpu('weights', [n_inputs, nodes], tf.truncated_normal_initializer(stddev=0.04))

        # create biases
        biases = _variable_on_cpu('biases', [nodes], tf.constant_initializer(0.1))

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
        biases = _variable_on_cpu('biases', [filter_num], tf.constant_initializer(0.1))
        preactivations = tf.nn.bias_add(conv, biases)

        # calculate activations
        activations = act(preactivations, name=scope.name)

        # create summary op
        _activation_summary(activations)

        return activations


class LayerAbstract(object):
    pass

    @abstractmethod
    def process(self, input_tensor):
        """
        Processes input tensor, i.e. applies all operations that layer has
        :param input_tensor: tensor, input data
        :return: list of tensors of length >= 1
        """
        pass


class LayerFullyConnected(LayerAbstract):

    def __init__(self, name, nodes, batch_size, act=tf.nn.relu, return_preactivations=False):
        self.name = name
        self.nodes = nodes
        self.activation_function = act
        self.batch_size = batch_size
        self.return_preactivations = return_preactivations

    def process(self, input_tensor):
        shape = input_tensor.get_shape()
        if shape.ndims == 4:
            n = shape[1] * shape[2] * shape[3]
            _input_tensor = tf.reshape(input_tensor, tf.pack([self.batch_size, -1]))
            _input_tensor.set_shape([None, n])
        else:
            _input_tensor = input_tensor

        return fc_nn_layer(self.name,
                           _input_tensor,
                           self.nodes,
                           self.activation_function,
                           self.return_preactivations)


class LayerConvolutional(LayerAbstract):

    def __init__(self, name, filter_size, filter_num, strides=[1, 1], act=tf.nn.relu):
        """

        :param name: string, layer name
        :param filter_size: list of ints with length = 2, height and width of filter
        :param filter_num: int, number of output channels, or number of filters
        :param strides: list of ints with length = 2, steps in each direction. [1, 1] by default, i.e. whole image is used
        :param act: tf.nn activation function, relu by default
        """
        self.name = name
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.strides = strides
        self.act = act

    def process(self, input_tensor):
        return conv_layer(self.name,
                          input_tensor,
                          self.filter_size,
                          self.filter_num,
                          self.strides,
                          self.act)


class LayerPooling(LayerAbstract):

    def __init__(self, name, size, strides):
        """

        :param name: string
        :param size: list of int with length = 2
        :param strides: list of int with length = 2
        """
        self.name = name
        self.size = size
        self.strides = strides

    def process(self, input_tensor):
        ksize = [1] + self.size + [1]
        strides = [1] + self.strides + [1]
        return tf.nn.max_pool(input_tensor,
                              ksize=ksize,
                              strides=strides,
                              padding='SAME',
                              name=self.name)


class LayerNormalization(LayerAbstract):

    def __init__(self, name, depth_radius):
        """

        :param name: string
        :param radius: int
        """
        self.name = name
        self.depth_radius = depth_radius

    def process(self, input_tensor):
        return tf.nn.local_response_normalization(input_tensor,
                                                  self.depth_radius,
                                                  name=self.name)

