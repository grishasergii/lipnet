from __future__ import division
import tensorflow as tf
import model
import os.path
from . import FLAGS
from evaluate import evaluate


def _prepare_dir(directory):
    """
    Prepares a directory. If it exists everything is deleted, otherwise the directory is created
    :param directory: string, path to directory to be emptied and/or created
    :return: nothing
    """
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)


def train(dataset, model, validation_set=None):
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
                if validation_set is None:
                    loss, acc = sess.run([model.cost, model.accuracy], feed_dict={model.x: batch_x,
                                                                                  model.y: batch_y,
                                                                                  model.keep_prob: 1.0})
                    print "Training: step: {} of {} loss: {:.6f} accuracy: {:.4f}".format(step, total_steps, loss, acc)
                else:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    model.saver.save(sess, checkpoint_path, global_step=step)
                    evaluate(validation_set, model, do_restore=True)
                    validation_set.reset()

            if step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=step)
            step += 1
            batch = dataset.next_batch()

