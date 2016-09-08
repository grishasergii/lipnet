import tensorflow as tf
from layer import *
from lipnet_architecture import *
from lipnet_tf import FLAGS


class Model(object):

    def __init__(self, num_classes, layer_definition):
        tf.reset_default_graph()
        self.dropout = 0.75

        # tf Graph input
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='Images')
            self.y = tf.placeholder(tf.float32, [None, num_classes], name='Labels')

        with tf.name_scope('Parameters'):
            with tf.name_scope('Learning_rate') as scope:
                self.learning_rate = tf.placeholder(tf.float32, name='Learning_rate')
                tf.scalar_summary(scope, self.learning_rate)
            with tf.name_scope('Dropout') as scope:
                self.keep_prob = tf.placeholder(tf.float32, name='Keep_probability')
                tf.scalar_summary(scope, self.keep_prob)

        logits = self._get_logits(layer_definition)
        self.predictions = tf.nn.softmax(logits)

        one_hot_pred = tf.argmax(self.predictions, 1)
        correct_pred = tf.equal(one_hot_pred, tf.argmax(self.y, 1))

        with tf.name_scope('Performance_Batch'):
            with tf.name_scope('Accuracy') as scope:
                # accuracy_batch
                self.accuracy_batch = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.scalar_summary(scope, self.accuracy_batch)
            with tf.name_scope('Cross_entropy') as scope:
                # loss_batch function
                self.loss_batch = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))
                tf.scalar_summary(scope, self.loss_batch)
            #"""
            with tf.name_scope('Weights_loss') as scope:
                weight_loss_list = tf.get_collection('weight_losses')
                if len(weight_loss_list) > 0:
                    self.loss_weights = tf.add_n()
                else:
                    self.loss_weights = 0
                tf.scalar_summary(scope, self.loss_weights)
            with tf.name_scope('Total_loss') as scope:
                self.loss_total = tf.add(self.loss_batch, self.loss_weights)
                tf.scalar_summary(scope, self.loss_total)
            #"""

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_total)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

    def _get_logits(self, layer_definitions):
        layer = self.x
        for ld in layer_definitions:
            if ld.layer_type == LayerEnum.Convolutional:
                layer = LayerConv2d.apply(ld.name, layer, ld.filter_size, ld.filter_num, ld. stride)
            elif ld.layer_type == LayerEnum.PoolingMax:
                layer = LayerMaxPool.apply(ld.name, layer, ld.pooling_size, ld.stride)
            elif ld.layer_type == LayerEnum.FullyConnected:
                act = {
                    ActivationFunctionEnum.Relu: tf.nn.relu,
                    ActivationFunctionEnum.Sigmoid: tf.nn.sigmoid,
                    ActivationFunctionEnum.Softmax: tf.nn.softmax
                }[ld.activation_function]
                layer = LayerFullyConnected.apply(ld.name, layer, ld.fc_nodes, act, self.keep_prob)
            elif ld.layer_type == LayerEnum.Output:
                layer = LayerOutput.apply(ld.name, layer, ld.fc_nodes)

        out = layer
        return out






















