from __future__ import division
import tensorflow as tf
import model
from datetime import datetime
import contextlib
import numpy as np
import os.path
from . import FLAGS
import pickle

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


def train(train_set, validation_set, layer_definitions):
    """
    Train lipnet CNN with Tensorflow
    :param train_set: object that implements DatasetAbstract contract
    :param path_to_images: path to folder with images
    :return:
    """

    if validation_set is not None:
        assert train_set.get_num_classes() == validation_set.get_num_classes(),\
            "Number of classes in train and validation sets must be the same"
    num_classes = train_set.get_num_classes()
    # Prepare output directories. Empty them if exist, otherwise create
    _prepare_dir(FLAGS.log_train_dir)
    _prepare_dir(FLAGS.checkpoint_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # get images and labels, training set

        images = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='images_input')
        labels = tf.placeholder(tf.float32, [None, train_set.get_num_classes()], name='labels_input')
        batch_size = tf.placeholder(tf.int32, name='batch_size')

        # Build a graph that computes the logits predictions.
        # predictions - predicted probabilities of belonging to all classes
        logits, predictions = model.get_predictions(images, batch_size, layer_definitions)

        # calculate loss
        loss = model.get_loss(logits, labels)
        accuracy = model.get_accuracy(predictions, labels)

        tf.scalar_summary('Loss', loss)

        # Build a Graph that trains the model with one batch of examples
        # and updates the model parameters
        train_op = model.train(loss)

        # Create a saver
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on th TF collection of summaries
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below
        init = tf.initialize_all_variables()

        # Start running operations on the Graph
        sess = tf.Session()
        sess.run(init)

        # Create a summary writer
        summary_writer = tf.train.SummaryWriter(FLAGS.log_train_dir, sess.graph)
        max_steps = 201
        format_str = '%s: step %d of %d: %s, loss = %.4f, accuracy = %.4f'
        batch = train_set.next_batch()
        total_steps = train_set.num_steps
        step = 0
        while batch is not None:
            step += 1
            # perform training
            _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict={images: batch.images,
                                                                                 labels: batch.labels,
                                                                                 batch_size: batch.size})
            print format_str % (datetime.now(), step, total_steps, 'training', loss_value, acc)

            summary_str = sess.run(summary_op, feed_dict={images: batch.images,
                                                          labels: batch.labels,
                                                          batch_size: batch.size})
            summary_writer.add_summary(summary_str, step)
            batch = train_set.next_batch()

            if validation_set is not None and step % 10 == 0:
                # perform evaluation on validation set
                print '%s: ...validating' % (datetime.now())
                validation_set.reset()
                batch_validation = validation_set.next_batch()
                loss_value = acc = 0
                while batch_validation is not None:
                    loss_value_batch, acc_batch = sess.run([loss, accuracy],
                                                           feed_dict={images: batch_validation.images,
                                                                      labels: batch_validation.labels,
                                                                      batch_size: batch_validation.size})

                    loss_value += loss_value_batch * batch_validation.size
                    acc += acc_batch * batch_validation.size
                    batch_validation = validation_set.next_batch()
                loss_value /= validation_set.get_count()
                acc /= validation_set.get_count()
                print format_str % (datetime.now(), step, total_steps, 'validation', loss_value, acc)

            if step % 100 == 0:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # save final checkpoint
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

        # save layer definitions
        layer_def_path = os.path.join(FLAGS.checkpoint_dir, 'layer_definitions.pickle')
        with open(layer_def_path, 'wb') as handle:
            pickle.dump(layer_definitions, handle)