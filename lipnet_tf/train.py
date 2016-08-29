from __future__ import division, print_function
from . import FLAGS
import tensorflow as tf
import os.path
from evaluate import evaluate
from datetime import datetime
import numpy as np
import math
from collections import OrderedDict


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

def train(dataset, model, validation_set=None, verbose=True, eval_step=None):
    step = 0
    total_steps = dataset.num_steps
    prepare_dir(os.path.join(FLAGS.logdir, 'train'))

    merged = tf.merge_all_summaries()

    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
        validation_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'validation'), sess.graph)

        sess.run(model.init)

        batch = dataset.next_batch()
        while batch is not None:
            step += 1
            if verbose:
                print("\r{}: Training step {} of {}".format(datetime.now(), step, total_steps), end='')

            batch_x = batch.images
            batch_y = batch.labels

            sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                 model.y: batch_y,
                                                 model.keep_prob: model.dropout,
                                                 model.learning_rate: 0.001})

            if eval_step is not None and step % eval_step == 0:
                if verbose:
                    print("\r{}: step: {} of {} ".format(datetime.now(), step, total_steps), end='')
                # Do evaluation on training set and write summaries
                summary, batch_loss, batch_acc = sess.run([merged, model.cost, model.accuracy], feed_dict={model.x: batch_x,
                                                                              model.y: batch_y,
                                                                              model.keep_prob: 1.0,
                                                                                               model.learning_rate: 0.001})
                if verbose:
                    print("\r{}: step: {} of {} Training:\t batch loss: {:.6f} accuracy: {:.4f}".
                          format(datetime.now(), step, total_steps, batch_loss, batch_acc), end='')
                train_writer.add_summary(summary, step)

                # Do evaluation on validation set and write summaries
                if validation_set is not None:
                    _evaluate(sess, model, validation_set, verbose=verbose)

            if step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=step)

            batch = dataset.next_batch()

        if validation_set is not None:
            validation_loss, validation_acc, validation_cf = _evaluate(sess, model, validation_set, verbose=verbose)


        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=step)

    if verbose:
        print('')
        print('{}: Training finished'.format(datetime.now()))

    output = {
        #"final_batch_loss": "%.4f" % batch_loss,
        #"final_batch_acc": "%.4f" % batch_acc,
    }
    train_stats = None
    validation_stats = None
    if validation_set is not None:
        validation_stats = {}
        validation_stats['loss'] = validation_loss
        validation_stats['acc'] = validation_acc
        validation_stats['cf'] = validation_cf

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