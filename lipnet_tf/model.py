import tensorflow as tf
from layer import *
from lipnet_architecture import *


def _log_loss(y, p):
    """
    Multi class log loss
    :param y: tensor, represents true classes
    :param p: tensor, represent predictions
    :return: scalar
    """
    return - tf.reduce_mean(tf.mul(y, tf.log(p)))


def _loss_summary(x):
    """
    Add summaries for losses
    :param x: total loss from get_loss()
    :return:
    """

def _get_layer_from_definition(layer_definition):
    """
    Transforms layer definition to layer object
    :param layer_definition: LayerDefinition named tuple
    :return: Layer* object
    """
    act = None
    if layer_definition.activation_function is not None:
        act = {
            ActivationFunctionEnum.Relu: tf.nn.relu,
            ActivationFunctionEnum.Sigmoid: tf.nn.sigmoid,
            ActivationFunctionEnum.Softmax: tf.nn.softmax
        }[layer_definition.activation_function]

    if layer_definition.layer_type == LayerEnum.Convolutional:
        return LayerConvolutional(layer_definition.name,
                                   layer_definition.filter_size,
                                   layer_definition.filter_num,
                                   layer_definition.strides,
                                   act)
    elif layer_definition.layer_type == LayerEnum.FullyConnected:
        return LayerFullyConnected(layer_definition.name,
                                    layer_definition.fc_nodes,
                                    act,
                                    layer_definition.return_preactivations)
    elif layer_definition.layer_type == LayerEnum.PoolingMax:
        return LayerPooling(layer_definition.name,
                             layer_definition.pooling_size,
                             layer_definition.strides)
    elif layer_definition.layer_type == LayerEnum.Normalization:
        return LayerNormalization(layer_definition.name, layer_definition.depth_radius)

    return None

def get_predictions(images, layer_definitions):
    """
    Build the lipnet model, feed images and get predictions of classes
    :param images: images to classify
    :return: predictions (logits)
    """
    # Instantiate all variables using tf.get_variable() instead of tf.Variable()
    # if you want to train lipnet on multiple GPU.

    layers = []
    for ld in layer_definitions:
        layers.append(_get_layer_from_definition(ld))

    input_tensor = images
    for layer in layers[:-1]:
        input_tensor = layer.process(input_tensor)

    logits, softmax = layers[-1].process(input_tensor)

    return logits, softmax


def get_loss(predictions, labels):
    """
    Calculate loss
    :param predictions: tensor
    :param labels: tensor
    :return: tensor
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, labels,
                                                                   name='cross_entropy_per_example')

    loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #loss = _log_loss(labels, predictions)
    return loss
    #tf.add_to_collection('losses', loss)
    #return tf.add_n(tf.get_collection('losses'), name='total_loss')


def get_accuracy(predictions, labels):
    """
    Returns accuracy - fraction of correct predictions
    :param predictions: tensor, predicted probabilities
    :param labels: tensor, one-hot encoded labels
    :return: scalar
    """
    correct_prediction = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, layer_definitions, dropout):

    layer = x
    last_filter_num = 1
    for ld in layer_definitions:
        if ld.layer_type == LayerEnum.Convolutional:
            weights = tf.Variable(tf.random_normal(ld.filter_size + [last_filter_num, ld.filter_num]))
            last_filter_num = ld.filter_num
            biases = tf.Variable(tf.random_normal([ld.filter_num]))
            layer = conv2d(layer, weights, biases)
        elif ld.layer_type == LayerEnum.PoolingMax:
            layer = maxpool2d(layer, ld.pooling_size[0])
        elif ld.layer_type == LayerEnum.FullyConnected:
            shape = layer.get_shape()
            if shape.ndims == 4:
                n = shape[1] * shape[2] * shape[3]
                layer = tf.reshape(layer, tf.pack([-1, n]))
                layer.set_shape([None, n])
            n = layer.get_shape()[1].value
            weights = tf.Variable(tf.random_normal([n, ld.fc_nodes]))
            biases = tf.Variable(tf.random_normal([ld.fc_nodes]))
            layer = tf.add(tf.matmul(layer, weights), biases)
            act = {
                ActivationFunctionEnum.Relu: tf.nn.relu,
                ActivationFunctionEnum.Sigmoid: tf.nn.sigmoid,
                ActivationFunctionEnum.Softmax: tf.nn.softmax
            }[ld.activation_function]
            layer = act(layer)
            layer = tf.nn.dropout(layer, dropout)
        elif ld.layer_type == LayerEnum.Output:
            shape = layer.get_shape()
            if shape.ndims == 4:
                n = shape[1] * shape[2] * shape[3]
                layer = tf.reshape(layer, tf.pack([-1, n]))
                layer.set_shape([None, n])
            n = layer.get_shape()[1].value
            weights = tf.Variable(tf.random_normal([n, ld.fc_nodes]))
            biases = tf.Variable(tf.random_normal([ld.fc_nodes]))
            layer = tf.add(tf.matmul(layer, weights), biases)

    """
    weights = {
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 32])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 3]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([3])),
    }

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    """
    out = layer
    return out




























