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

def _get_layer_from_definition(layer_definition, batch_size):
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
                                    batch_size,
                                    act,
                                    layer_definition.return_preactivations)
    elif layer_definition.layer_type == LayerEnum.PoolingMax:
        return LayerPooling(layer_definition.name,
                             layer_definition.pooling_size,
                             layer_definition.strides)
    elif layer_definition.layer_type == LayerEnum.Normalization:
        return LayerNormalization(layer_definition.name, layer_definition.depth_radius)

    return None

def get_predictions(images, batch_size, num_classes, layer_definitions):
    """
    Build the lipnet model, feed images and get predictions of classes
    :param images: images to classify
    :return: predictions (logits)
    """
    # Instantiate all variables using tf.get_variable() instead of tf.Variable()
    # if you want to train lipnet on multiple GPU.

    layers = []
    for ld in layer_definitions:
        layers.append(_get_layer_from_definition(ld, batch_size))

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


def get_batch_size(images, labels):
    """
    Returns number of examples in a batch
    :param images: tensor
    :param labels: tensor
    :return: scalar
    """
    num_images = tf.shape(images)[0]
    num_labels = tf.shape(labels)[0]
    #assert num_images == num_labels, 'Number of images and corresponding labels must be the same'
    return num_images


def train(total_loss):
    """

    :param total_loss:
    :param global_step:
    :return:
    """

    lr = 0.1
    #train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(total_loss)
    """
    with tf.control_dependencies([total_loss]):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    """
    return train_op


























