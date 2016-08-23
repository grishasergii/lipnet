from __future__ import division
from . import FLAGS
import tensorflow as tf
import os.path
from evaluate import evaluate
import datetime
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


def train(dataset, model, validation_set=None, verbose=True):
    step = 0
    total_steps = dataset.num_steps
    eval_step = 10
    prepare_dir(os.path.join(FLAGS.logdir, 'train'))

    merged = tf.merge_all_summaries()

    tf.logging.set_verbosity(tf.logging.ERROR)

    # 0 - loss
    # 1 - accuracy
    # 2 - confusion matrix

    validation_stats = np.zeros([int(math.floor(total_steps / eval_step)), 3])
    validation_cf = np.zeros([int(math.floor(total_steps / eval_step)),
                              validation_set.get_num_classes() * validation_set.get_num_classes()])

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
        validation_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'validation'), sess.graph)

        sess.run(model.init)

        batch = dataset.next_batch()
        while batch is not None:
            step += 1

            batch_x = batch.images
            batch_y = batch.labels

            sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                 model.y: batch_y,
                                                 model.keep_prob: model.dropout,
                                                 model.learning_rate: 0.001})

            if step % eval_step == 0:
                if verbose:
                    print "{}: step: {} of {} ".format(datetime.datetime.now(), step, total_steps)
                # Do evaluation on training set and write summaries
                summary, batch_loss, batch_acc = sess.run([merged, model.cost, model.accuracy], feed_dict={model.x: batch_x,
                                                                              model.y: batch_y,
                                                                              model.keep_prob: 1.0,
                                                                                               model.learning_rate: 0.001})
                if verbose:
                    print "Training:\t batch loss: {:.6f} accuracy: {:.4f}".format(batch_loss, batch_acc)
                train_writer.add_summary(summary, step)

                # Do evaluation on validation set and write summaries
                if validation_set is not None:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step=step)
                    validation_set.reset()
                    validation_loss, validation_acc, confusion_matrix = evaluate(validation_set, model, do_restore=True, verbose=False)
                    if verbose:
                        print "Validation:\t total loss: {:.6f} accuracy: {:.4f}".format(validation_loss, validation_acc)
                    i = int(math.floor(step / eval_step)) - 1
                    validation_stats[i, 0] = validation_loss
                    validation_stats[i, 1] = validation_acc
                    validation_cf[i, :] = ["%.2f" % f for f in confusion_matrix.normalized.flatten()]


            if step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=step)

            batch = dataset.next_batch()

        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=step)
        if verbose:
            print 'Training finished'

        min_validation_loss_i = validation_stats.argmin(axis=0)[0]
        max_validation_acc_i = validation_stats.argmax(axis=0)[1]
        output = {
            "final_batch_loss": "%.4f" % batch_loss,
            "final_batch_acc": "%.4f" % batch_acc,
            "z_confusion_matrix_final": ["%.2f" % x for x in validation_cf[-1, :]],
            "min_validation_loss:": "%.4f" % validation_stats[min_validation_loss_i, 0],
            "max_validation_acc": "%.4f" % validation_stats[max_validation_acc_i, 1],
            "min_validation_loss_epoch": dataset.step_to_epoch((min_validation_loss_i + 1) * eval_step),
            "z_confusion_matrix_min_loss": ["%.2f" % x for x in validation_cf[min_validation_loss_i, :]],
            "max_validation_acc_epoch": dataset.step_to_epoch((max_validation_acc_i + 1) * eval_step),
            "z_confusion_matrix_max_acc": ["%.2f" % x for x in validation_cf[max_validation_acc_i, :]]
        }

        return OrderedDict(sorted(output.items(), key=lambda t: t[0]))