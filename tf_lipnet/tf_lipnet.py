import tensorflow as tf


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

    # output layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_on_cpu('weights', tf.random_normal([384, NUM_CLASSES]))
        biases = _variable_on_cpu('biases', tf.constant(0.1, shape=[NUM_CLASSES]))
        softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)

    return softmax_linear

def get_loss(predictions, labels):
    """
    Calculate loss
    :param predictions: tensor
    :param labels: tensor
    :return: tensor
    """
    #labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, labels,
                                                                   name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step, batch_size):
    """

    :param total_loss:
    :param global_step:
    :return:
    """
    num_batches_per_epoch = 2000 / batch_size
    lr = 0.1
    with tf.control_dependencies([total_loss]):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


























