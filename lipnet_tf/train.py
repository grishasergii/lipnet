from __future__ import division
import tensorflow as tf
import model
from datetime import datetime
import contextlib
import numpy as np
import os.path
from . import FLAGS
import pickle
import confusion_matrix as cf

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(original)


def _prepare_dir(directory):
    """
    Prepares a directory. If it exists everything is deleted, otherwise the directory is created
    :param directory: string, path to directory to be emptied and/or created
    :return: nothing
    """
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)


class Model(object):

    def __init__(self, num_classes, layer_definition):
        self.learning_rate = 0.01
        self.dropout = 0.75

        # Prepare output directories. Empty them if exist, otherwise create
        #_prepare_dir(FLAGS.log_train_dir)
        #_prepare_dir(FLAGS.checkpoint_dir)

        # tf Graph input
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='Images')
            self.y = tf.placeholder(tf.float32, [None, num_classes], name='Labels')
        self.keep_prob = tf.placeholder(tf.float32)

        logits = model.conv_net(self.x, layer_definition, self.keep_prob)
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


def train(dataset, model):
    step = 0
    total_steps = dataset.num_steps
    with tf.Session() as sess:
        sess.run(model.init)
        # training
        batch = dataset.next_batch()
        while batch is not None:
            batch_x = batch.images
            batch_y = batch.labels

            sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                 model.y: batch_y,
                                                 model.keep_prob: model.dropout})

            if step % 10 == 0:
                loss, acc = sess.run([model.cost, model.accuracy], feed_dict={model.x: batch_x,
                                                                              model.y: batch_y,
                                                                              model.keep_prob: 1.0})
                print "Training: step: {} of {} loss: {:.6f} accuracy: {:.4f}".format(step, total_steps, loss, acc)

            if step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=step)
            step += 1
            batch = dataset.next_batch()


def evaluate(dataset, model, do_restore=True):
    with tf.Session() as sess:
        if do_restore:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                model.saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print 'Warning: no checkpoint found'
        loss = 0
        acc = 0
        examples_count = 0
        batch = dataset.next_batch()
        while batch is not None:
            batch_x = batch.images
            batch_y = batch.labels
            batch_loss, batch_acc, batch_pred = sess.run([model.cost, model.accuracy, model.predictions],
                                                         feed_dict={model.x: batch_x,
                                                                    model.y: batch_y,
                                                                    model.keep_prob: 1.0})
            print "Evaluating: loss: {:.6f} accuracy: {:.4f}".format(batch_loss, batch_acc)
            dataset.set_predictions(batch.ids, batch_pred)
            loss += batch_loss * batch.size
            acc += batch_acc * batch.size
            examples_count += batch.size
            batch = dataset.next_batch()

        loss /= examples_count
        acc /= examples_count

        print 'Total loss: {:.4f} accuracy: {:.4f}'.format(loss, acc)
        dataset.evaluate()


def train_simple(dataset, layer_definitions, do_train=True, do_evaluate=False):
    learning_rate = 0.01
    dropout = 0.75

    # Prepare output directories. Empty them if exist, otherwise create
    _prepare_dir(FLAGS.log_train_dir)
    if do_train:
        _prepare_dir(FLAGS.checkpoint_dir)

    # tf Graph input
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='Images')
        y = tf.placeholder(tf.float32, [None, dataset.get_num_classes()], name='Labels')
    keep_prob = tf.placeholder(tf.float32)

    logits = model.conv_net(x, layer_definitions, keep_prob)
    predictions = tf.nn.softmax(logits)

    # accuracy
    one_hot_pred = tf.argmax(logits, 1)
    correct_pred = tf.equal(one_hot_pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    step = 0
    total_steps = dataset.num_steps

    with tf.Session() as sess:
        sess.run(init)
        # training
        if do_train:
            batch = dataset.next_batch()
            while batch is not None:
                batch_x = batch.images
                batch_y = batch.labels

                sess.run(optimizer, feed_dict={x: batch_x,
                                               y: batch_y,
                                               keep_prob: dropout})

                if step % 10 == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                      y: batch_y,
                                                                      keep_prob: 1.0})
                    print "Training: step: {} of {} loss: {:.6f} accuracy: {:.4f}".format(step, total_steps, loss, acc)

                if step % 100 == 0:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                step += 1
                batch = dataset.next_batch()
            dataset.reset()

        # evaluating
        if do_evaluate:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print 'Warning: no checkpoint found'
            loss = 0
            acc = 0
            examples_count = 0
            batch = dataset.next_batch()
            while batch is not None:
                batch_x = batch.images
                batch_y = batch.labels
                batch_loss, batch_acc, batch_pred = sess.run([cost, accuracy, predictions],
                                                             feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 1.0})
                print "Evaluating: loss: {:.6f} accuracy: {:.4f}".format(batch_loss, batch_acc)
                dataset.set_predictions(batch.ids, batch_pred)
                loss += batch_loss * batch.size
                acc += batch_acc * batch.size
                examples_count += batch.size
                batch = dataset.next_batch()

            loss /= examples_count
            acc /= examples_count

            print 'Total loss: {:.4f} accuracy: {:.4f}'.format(loss, acc)
            dataset.evaluate()

