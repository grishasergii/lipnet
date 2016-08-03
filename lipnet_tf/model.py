import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


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


def _fc_nn_layer(name, input_tensor, nodes, act=tf.nn.relu, return_preactivations=False):
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
        activations =act(preactivations, name=scope.name)

        # create summary op
        _activation_summary(activations)

        if return_preactivations:
            return preactivations, activations

        return activations



def _conv_layer(name, input_tensor, filter_size, filter_num, stride=[1, 1], act=tf.nn.relu):
    """
    Reusable code for making a convolutional neural net layer.
    Good summary of what convolutional layer is
    http://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
    :param name: string, layer name, used for summaries and improves readability of tensorboard
    :param input_tensor: tensor, input to the layer
    :param filter_size: 1 x 2 tensor, height and width of filter
    :param filter_num: int, number of output channels, or number of filters
    :param stride: 1 x 2 tensor, steps in each direction. [1, 1] by default, i.e. whole image is used
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


def get_predictions(images, batch_size, num_classes):
    """
    Build the lipnet model, feed images and get predictions of classes
    :param images: images to classify
    :return: predictions (logits)
    """
    # Instantiate all variables using tf.get_variable() instead of tf.Variable()
    # if you want to train lipnet on multiple GPU.
    #
    # First convolutional layer conv1
    filter_num = 64

    conv1 = _conv_layer('conv1',
                        images,
                        [5, 5],
                        filter_num)

    # Pooling layer: maxpool
    pooling_size = 2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, pooling_size, pooling_size, 1], strides=[1, pooling_size, pooling_size, 1],
                           padding='SAME', name='pool1')

    # Normalization
    norm1 = tf.nn.local_response_normalization(pool1, 4, name="norm1")

    conv2 = _conv_layer('conv2',
                        norm1,
                        [5, 5],
                        filter_num)

    norm_size = 4
    norm2 = tf.nn.local_response_normalization(conv2, norm_size, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, pooling_size, pooling_size, 1],
                           strides=[1, pooling_size, pooling_size, 1],
                           padding='SAME',
                           name='pool2')

    # fully connected layer
    #batch_size = images.get_shape()[0]

    # an ugly hack to calculate length of flattened image after convolution and max pooling applied
    # it is done because tensorflow does not handle tensors of dynamic size in easy way
    n = FLAGS.image_width * FLAGS.image_height * filter_num / \
        (pooling_size * pooling_size * norm_size)
    reshape = tf.reshape(pool2, tf.pack([batch_size, -1]))
    reshape.set_shape([None, n])

    fc1 = _fc_nn_layer('fc1',
                       reshape,
                       384)

    fc2 = _fc_nn_layer('fc2',
                       fc1,
                       192)

    logits, softmax = _fc_nn_layer('softmax_linear',
                                   fc2,
                                   num_classes,
                                   act=tf.nn.softmax,
                                   return_preactivations=True)

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


def train(total_loss, global_step, batch_size):
    """

    :param total_loss:
    :param global_step:
    :return:
    """
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
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


























