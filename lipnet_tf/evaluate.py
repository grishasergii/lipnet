from __future__ import division, print_function
import tensorflow as tf
from . import FLAGS
from datetime import datetime
from model import Model
import os

def evaluate(dataset, model, session=None, do_restore=True, verbose=True):
    """

    :param dataset:
    :param model:
    :param do_restore:
    :return:
    """
    dataset.reset()
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
        batch = dataset.next_batch()
        i = 1
        total_batches = dataset.num_batches
        while batch is not None:
            print('\r{}: Evaluating batch {} of {}'.format(datetime.now(), i, total_batches), end='')
            batch_x = batch.images
            batch_y = batch.labels
            batch_loss, batch_acc, batch_pred = sess.run([model.cost, model.accuracy, model.predictions],
                                                         feed_dict={model.x: batch_x,
                                                                    model.y: batch_y,
                                                                    model.keep_prob: 1.0})
            dataset.set_predictions(batch.ids, batch_pred)
            loss += batch_loss * batch.size
            acc += batch_acc * batch.size
            examples_count += batch.size
            batch = dataset.next_batch()

            i += 1

        loss /= examples_count
        acc /= examples_count

        if verbose:
            print('')
            print('{}: Evaluation: total loss: {:.4f} accuracy: {:.4f}'.format(datetime.now(), loss, acc))
            #dataset.confusion_matrix.print_to_console()
    finally:
        if close_session:
            sess.close()

    return loss, acc, dataset.confusion_matrix
        #dataset.evaluate()