import tensorflow as tf
from layer import *
from lipnet_architecture import *
from . import FLAGS


class Model(object):

    def __init__(self, num_classes, layer_definition):
        self.learning_rate = 0.01
        self.dropout = 0.75

        # tf Graph input
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='Images')
            self.y = tf.placeholder(tf.float32, [None, num_classes], name='Labels')
        self.keep_prob = tf.placeholder(tf.float32)

        logits = self._get_logits(layer_definition)
        self.predictions = tf.nn.softmax(logits)

        # accuracy
        one_hot_pred = tf.argmax(logits, 1)
        correct_pred = tf.equal(one_hot_pred, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # cost function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

    def _get_logits(self, layer_definitions):
        layer = self.x
        for ld in layer_definitions:
            if ld.layer_type == LayerEnum.Convolutional:
                layer = LayerConv2d.apply(layer, ld.filter_size, ld.filter_num, ld. stride)
            elif ld.layer_type == LayerEnum.PoolingMax:
                layer = LayerMaxPool.apply(layer, ld.pooling_size, ld.stride)
            elif ld.layer_type == LayerEnum.FullyConnected:
                act = {
                    ActivationFunctionEnum.Relu: tf.nn.relu,
                    ActivationFunctionEnum.Sigmoid: tf.nn.sigmoid,
                    ActivationFunctionEnum.Softmax: tf.nn.softmax
                }[ld.activation_function]
                layer = LayerFullyConnected.apply(layer, ld.fc_nodes, act, self.keep_prob)
            elif ld.layer_type == LayerEnum.Output:
                layer = LayerOutput.apply(layer, ld.fc_nodes)

        out = layer
        return out






















