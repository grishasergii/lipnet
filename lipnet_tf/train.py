from __future__ import division
from . import FLAGS
import tensorflow as tf
import os.path
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

    _prepare_dir(os.path.join(FLAGS.logdir, 'train'))
    _prepare_dir(os.path.join(FLAGS.logdir, 'validation'))

    merged = tf.merge_all_summaries()

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'train'),
                                              sess.graph)

        sess.run(model.init)
        # training
        batch = dataset.next_batch()
        while batch is not None:
            batch_x = batch.images
            batch_y = batch.labels

            sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                 model.y: batch_y,
                                                 model.keep_prob: model.dropout,
                                                 model.learning_rate: 0.001})

            if step % 10 == 0:
                if validation_set is None:
                    summary, loss, acc = sess.run([merged, model.cost, model.accuracy], feed_dict={model.x: batch_x,
                                                                                  model.y: batch_y,
                                                                                  model.keep_prob: 1.0,
                                                                                                   model.learning_rate: 0.001})
                    print "Training: step: {} of {} loss: {:.6f} accuracy: {:.4f}".format(step, total_steps, loss, acc)
                    train_writer.add_summary(summary, step)
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

        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=step)
        print 'Training finished'