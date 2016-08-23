from __future__ import division
import tensorflow as tf
from . import FLAGS
from model import Model
import os

def evaluate(dataset, model, do_restore=True, verbose=True):
    """

    :param dataset:
    :param model:
    :param do_restore:
    :return:
    """

    with tf.Session() as sess:
        if do_restore:
            checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                model.saver.restore(sess, checkpoint.model_checkpoint_path)
                if verbose:
                    print 'Model has been restored from {}'.format(checkpoint.model_checkpoint_path)
            else:
                if verbose:
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
            dataset.set_predictions(batch.ids, batch_pred)
            loss += batch_loss * batch.size
            acc += batch_acc * batch.size
            examples_count += batch.size
            batch = dataset.next_batch()

        loss /= examples_count
        acc /= examples_count

        if verbose:
            print 'Evaluating: total loss: {:.4f} accuracy: {:.4f}'.format(loss, acc)
        return loss, acc, dataset.confusion_matrix
        #dataset.evaluate()