import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, initial_value):
    """
    Helper to create a Variable stored on CPU memory
    :param name: name of the Variable
    :param initial_value: tensor
    :return: Variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.Variable(name=name, initial_value=initial_value, dtype=dtype)
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
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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


def get_predictions(images, batch_size):
    """
    Build the lipnet model, feed images and get predictions of classes
    :param images: images to classify
    :return: predictions (logits)
    """
    # Instantiate all variables using tf.get_variable() instead of tf.Variable()
    # if you want to train lipnet on multiple GPU.
    #
    # First convolutional layer conv1
    NUM_CLASSES = 3


    with tf.variable_scope('conv1') as scope:
        # 5x5 convolution 1 input 64 outputs
        kernel = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', tf.random_normal([32]))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        # _activation_summary

    # Pooling layer: maxpool
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # fully connected layer
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool1, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_on_cpu('weights', tf.random_normal([dim, 384]))
        biases = _variable_on_cpu('biases', tf.constant(0.1, shape=[384]))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(fc1)

    # output layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_on_cpu('weights', tf.random_normal([384, NUM_CLASSES]))
        biases = _variable_on_cpu('biases', tf.constant(0.1, shape=[NUM_CLASSES]))
        softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
        #softmax_linear = tf.nn.softmax(preactivate)
        _activation_summary(softmax_linear)

    return softmax_linear

def get_loss(predictions, labels):
    """
    Calculate loss
    :param predictions: tensor
    :param labels: tensor
    :return: tensor
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, labels,
                                                                   name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #multi_class_log_loss = _log_loss(labels, predictions)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step, batch_size):
    """

    :param total_loss:
    :param global_step:
    :return:
    """
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    lr = 0.1
    with tf.control_dependencies([total_loss]):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


























