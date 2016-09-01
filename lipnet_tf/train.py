from __future__ import division, print_function
from . import FLAGS
import tensorflow as tf
import os.path
from evaluate import evaluate
from datetime import datetime
import numpy as np
import math
from collections import OrderedDict
import matplotlib.pyplot as plt


def prepare_dir(directory):
    """
    Prepares a directory. If it exists everything is deleted, otherwise the directory is created
    :param directory: string, path to directory to be emptied and/or created
    :return: nothing
    """
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)


def _evaluate(sess, model, dataset, verbose=True):
    return evaluate(dataset, model, session=sess, do_restore=False, verbose=verbose)


def train(dataset, model, epochs, validation_set=None, verbose=True, eval_step=None, intermediate_evaluation=False,
          plot_path=''):
    global_step = 0
    total_batches = dataset.batches_count
    prepare_dir(os.path.join(FLAGS.logdir, 'train'))
    prepare_dir(os.path.join(FLAGS.logdir, 'validation'))

    #merged = tf.merge_all_summaries()

    tf.logging.set_verbosity(tf.logging.ERROR)

    train_accuracy = [0] * epochs
    train_loss = [0] * epochs
    validation_accuracy = [0] * epochs
    validation_loss = [0] * epochs

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
        validation_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'validation'), sess.graph)

        sess.run(model.init)

        for epoch in xrange(epochs):
            # iterate over entire dataset in minibatches
            for i, batch in enumerate(dataset.iterate_minibatches(shuffle_=True)):
                global_step += 1
                if verbose:
                    print("\r{}: Epoch {}/{} Training batch {} of {}".format(datetime.now(),
                                                                             epoch + 1, epochs,
                                                                             i + 1, total_batches), end='')

                batch_x = batch.images
                batch_y = batch.labels

                sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                     model.y: batch_y,
                                                     model.keep_prob: model.dropout,
                                                     model.learning_rate: 0.001})

            # do evaluation
            if intermediate_evaluation:
                if verbose:
                    print('\r{}: Epoch {}/{} Evaluating Training set:'.format(datetime.now(),
                                                                              epoch + 1, epochs,), end='')
                    # Do evaluation on training set and write summaries
                train_loss[epoch], train_accuracy[epoch], _, train_summary = \
                    _evaluate(sess, model, dataset, verbose=False)
                train_writer.add_summary(train_summary, epoch)

                # Do evaluation on validation set and write summaries
                if validation_set is not None:
                    if verbose:
                        print('\r{}: Epoch {}/{} Evaluating Validation set:'.format(datetime.now(),
                                                                                    epoch + 1, epochs,), end='')
                    validation_loss[epoch], validation_accuracy[epoch], _, validation_summary = \
                        _evaluate(sess, model, validation_set, verbose=False)
                    validation_writer.add_summary(validation_summary, epoch)


            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
            model.saver.save(sess, checkpoint_path, global_step=epoch)

        if validation_set is not None:
            final_loss, final_acc, final_cf, _ = _evaluate(sess, model, validation_set, verbose=verbose)

        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=global_step)

        print('\rPlotting weights', end='')
        w_tensor = tf.get_default_graph().get_tensor_by_name('conv1_weights:0')
        plot_conv_weights(w_tensor, plot_path, sess, 'conv1')
        w_tensor = tf.get_default_graph().get_tensor_by_name('conv2_weights:0')
        plot_conv_weights(w_tensor, plot_path, sess, 'conv2')

    if verbose:
        print('\r{}: Training finished'.format(datetime.now()))

    output = {
        #"final_batch_loss": "%.4f" % loss_batch,
        #"final_batch_acc": "%.4f" % batch_acc,
    }
    train_stats = {}
    validation_stats = None
    if validation_set is not None:
        validation_stats = {}
        validation_stats['loss'] = final_loss
        validation_stats['acc'] = final_acc
        validation_stats['cf'] = final_cf
        if intermediate_evaluation:
            validation_stats['loss_series'] = validation_loss
            validation_stats['acc_series'] = validation_accuracy
            train_stats['loss_series'] = train_loss
            train_stats['acc_series'] = train_accuracy

    """
    if validation_set is not None and validation_stats is not None:
        min_validation_loss_i = validation_stats.argmin(axis=0)[0]
        max_validation_acc_i = validation_stats.argmax(axis=0)[1]
        output["z_confusion_matrix_final"] = ["%.2f" % x for x in validation_cf[-1, :]],
        output["min_validation_loss:"] = "%.4f" % validation_stats[min_validation_loss_i, 0],
        output["max_validation_acc"] = "%.4f" % validation_stats[max_validation_acc_i, 1],
        output["min_validation_loss_epoch"] = dataset.step_to_epoch((min_validation_loss_i + 1) * eval_step),
        output["z_confusion_matrix_min_loss"] = ["%.2f" % x for x in validation_cf[min_validation_loss_i, :]],
        output["max_validation_acc_epoch"] = dataset.step_to_epoch((max_validation_acc_i + 1) * eval_step),
        output["z_confusion_matrix_max_acc"] = ["%.2f" % x for x in validation_cf[max_validation_acc_i, :]]
    """
    return train_stats, validation_stats


def plot_conv_weights(weights, out_dir, session, name):
    """
    Plot weights of convolutional layer
    From https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    :param weights: Tensorflow op
    :param out_dir: string, folder where plot will be saved
    :param session: Tensorflow session
    :param name: string, tensor name, plots are named as tensor
    :return: nothing, plot is saved to out_dir
    """
    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    if w_min < 0 and w_max > 0:
        cmap = 'seismic'
        vmax = max([abs(w_min), w_max])
        vmin = -vmax
    else:
        cmap = 'Greys'
        vmin = w_min
        vmax = w_max

    num_filters = w.shape[3]
    grid_x, grid_y = _get_grid_dim(num_filters)

    fig, axes = plt.subplots(min([grid_x, grid_y]),
                             max([grid_x, grid_y]))

    for channel in xrange(w.shape[2]):
        for i, ax in enumerate(axes.flat):
            img = w[:, :, channel, i]
            ax.imshow(img, vmin=vmin, vmax=vmax, interpolation='nearest', cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(out_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def _get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in xrange(1, int(math.sqrt(n)) +1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)