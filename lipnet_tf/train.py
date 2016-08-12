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
        #images = tf.placeholder(tf.float32, [None, 11], name='images_input')
        labels = tf.placeholder(tf.float32, [None, train_set.get_num_classes()], name='labels_input')
        batch_size = tf.placeholder(tf.int32, name='batch_size')

        # Build a graph that computes the logits predictions.
        # predictions - predicted probabilities of belonging to all classes
        logits, predictions = model.get_predictions(images, layer_definitions)

        # calculate loss
        loss = model.get_loss(logits, labels)
        accuracy = model.get_accuracy(predictions, labels)

        tf.scalar_summary('Loss', loss)

        # Build a Graph that trains the model with one batch of examples
        # and updates the model parameters
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

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
        print 'Getting first batch...'
        batch = train_set.next_batch()
        print 'Done'
        total_steps = train_set.num_steps
        step = 0
        while batch is not None:
            step += 1
            # perform training
            _, loss_value, acc = sess.run([optimizer, loss, accuracy], feed_dict={images: batch.images,
                                                                                 labels: batch.labels,
                                                                                 batch_size: batch.size})
            print format_str % (datetime.now(), step, total_steps, 'training', loss_value, acc)

            summary_str = sess.run(summary_op, feed_dict={images: batch.images,
                                                          labels: batch.labels,
                                                          batch_size: batch.size})
            summary_writer.add_summary(summary_str, step)
            # batch = train_set.next_batch()

            if validation_set is not None and step % 10 == 0:
                # perform evaluation on validation set
                print '%s: ...evaluating validation set' % (datetime.now())
                validation_set.reset()
                batch_validation = validation_set.next_batch()
                loss_value = acc = 0
                while batch_validation is not None:
                    loss_value_batch, acc_batch, p = sess.run([loss, accuracy, predictions],
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


def train_simple(dataset, test_set, layer_definitions):
    learning_rate = 0.01
    dropout = 0.75

    # tf Graph input
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='Images')
        y = tf.placeholder(tf.float32, [None, dataset.get_num_classes()], name='Labels')
    keep_prob = tf.placeholder(tf.float32)

    pred = model.conv_net(x, layer_definitions, keep_prob)
    #pred, _ = model.get_predictions(x, layer_definitions)

    softmax = tf.nn.softmax(pred)
    one_hot_pred = tf.argmax(pred, 1)

    threshold = tf.constant(0.25, dtype=tf.float32)
    raw_prob = tf.greater_equal(softmax, threshold)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(one_hot_pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.initialize_all_variables()
    step = 0
    with tf.Session() as sess:
        sess.run(init)
        #print 'Training...'
        batch = dataset.next_batch()
        #while batch is not None:
        while step <= 100:
            batch_x = batch.images
            batch_y = batch.labels

            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: dropout})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.0})
            if step == 100:
                print "Training: step: {} loss: {:.6f} accuracy: {:.4f}".format(step, loss, acc)
            step += 1
            #batch = dataset.next_batch()

        """
        print 'Evaluating...'
        batch = test_set.next_batch()
        while batch is not None:
            batch_x = batch.images
            batch_y = batch.labels

            loss, acc, pr = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.0})

            print "Evaluation: loss: {:.6f} accuracy: {:.4f}".format(loss, acc)
            confusion_matrix = cf.ConfusionMatrix(pr,
                                                  batch_y)
            confusion_matrix.print_to_console()
            batch = test_set.next_batch()
        """

