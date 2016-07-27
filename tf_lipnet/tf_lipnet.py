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


def _fc_nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
    """
    Reusable code for making a simple fully connected neural net layer.
    It does a matrix multiply, bias add, and then uses act function (sigmoid by default).
    It also sets up name scoping so that the resultant graph is easy to read and adds a number
    of summary operations.
    :param input_tensor: tensor, input to the layer
    :param input_dim: tensor
    :param output_dim: tensor
    :param layer_name: string
    :param act: tf.nn activation function
    :return: tensor, activations
    """
    #with tf.name_scope(layer_name):


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
        kernel = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=5e-2))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        # _activation_summary


    # Pooling layer: maxpool
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # Normalization
    norm1 = tf.nn.local_response_normalization(pool1, 4, name="norm1")

    # fully connected layer
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool1, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_on_cpu('weights', [dim, 384], tf.truncated_normal_initializer(stddev=0.04))
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        preactivations = tf.add(tf.matmul(reshape, weights), biases)
        fc1 = tf.nn.relu(preactivations, name=scope.name)
        _activation_summary(fc1)

    # output layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_on_cpu('weights', [384, NUM_CLASSES], tf.truncated_normal_initializer(stddev=0.04))
        biases_sf = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc1, weights), biases_sf, name=scope.name)
        softmax = tf.nn.softmax(logits)
        #_activation_summary(logits)

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


























