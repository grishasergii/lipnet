from __future__ import division, print_function
import tensorflow as tf
from . import FLAGS
from datetime import datetime
from model import Model
import os
import numpy as np


def evaluate(dataset, model, session=None, do_restore=True, verbose=True, return_confusion_matrix=False, summary_writer=None):
    """

    :param dataset:
    :param model:
    :param do_restore:
    :return:
    """

    if session is None:
        sess = tf.Session()
        close_session = True
    else:
        close_session = False
        sess = session
    try:
        if do_restore:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                model.saver.restore(sess, checkpoint.model_checkpoint_path)
                if verbose:
                    print('Model has been restored from {}'.format(checkpoint.model_checkpoint_path))
            else:
                if verbose:
                    print('Warning: no checkpoint found')

        loss = 0
        acc = 0
        examples_count = 0
        total_batches = dataset.batches_count

        merged = tf.merge_all_summaries()
        #merged = tf.merge_summary([summary_accuracy_total,
        #                           summary_loss_total])

        for i, batch in enumerate(dataset.iterate_minibatches(shuffle_=False)):
            if verbose:
                print('\r{}: Evaluating batch {} of {}'.format(datetime.now(), i + 1, total_batches), end='')
            batch_x = batch.images
            batch_y = batch.labels
            summary, batch_loss, batch_acc, batch_pred = sess.run([merged, model.loss_batch, model.accuracy_batch, model.predictions],
                                                         feed_dict={model.x: batch_x,
                                                                    model.y: batch_y,
                                                                    model.keep_prob: 1.0,
                                                                    model.learning_rate: 0})
            if return_confusion_matrix:
                dataset.set_predictions(batch.ids, batch_pred)
            loss += batch_loss * batch.size
            acc += batch_acc * batch.size
            examples_count += batch.size
            if summary_writer is not None:
                summary_writer.add_summary(summary, FLAGS.global_step)
                FLAGS.global_step += 1

        loss /= examples_count
        acc /= examples_count

        if verbose:
            print('')
            print('{}: total loss: {:.4f} accuracy_batch: {:.4f}'.format(datetime.now(), loss, acc))
            #dataset.confusion_matrix.print_to_console()
    finally:
        if close_session:
            sess.close()

    cf = None
    if return_confusion_matrix:
        cf = dataset.confusion_matrix

    return loss, acc, cf, summary
        #dataset.evaluate()