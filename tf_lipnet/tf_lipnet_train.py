import tensorflow as tf
import tf_lipnet_input
import tf_lipnet
from datetime import datetime
import contextlib
import numpy as np
import os.path

FLAGS = tf.app.flags.FLAGS

# Directory for train information output. It will be emptied at the beginning of every run
tf.app.flags.DEFINE_string('log_dir', './output/log', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './output/checkpoint', """Directory where to write checkpoint""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Maximum number of training epochs""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement""")


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


def train(particles_df, path_to_images, max_steps):
    """
    Train lipnet CNN with Tensorflow
    :param particles_df: pandas data frame that describes all particles
    :param path_to_images: path to folder with images
    :return:
    """
    # Prepare output directories. Empty them if exist, otherwise create
    _prepare_dir(FLAGS.log_dir)
    _prepare_dir(FLAGS.checkpoint_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # batch size. pass it as parameter later
        batch_size = 500

        # get images and labels, training set
        images, labels = tf_lipnet_input.inputs(particles_df,
                                                path_to_images,
                                                batch_size=batch_size)
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        # Build a graph that computes the logits predictions.
        # predictions - predicted probabilities of belonging to any of classes
        logits, predictions = tf_lipnet.get_predictions(images, batch_size=batch_size)

        # calculate loss
        loss = tf_lipnet.get_loss(logits, labels)
        accuracy = tf_lipnet.get_accuracy(predictions, labels)

        tf.scalar_summary('Loss', loss)

        # Build a Graph that trains the model with one batch of examples
        # and updates the model parameters
        train_op = tf_lipnet.train(loss, global_step, batch_size)

        # Create a saver
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on th TF collection of summaries
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below
        init = tf.initialize_all_variables()

        # Start running operations on the Graph
        sess = tf.Session()
        sess.run(init)

        # Start the queue runners
        tf.train.start_queue_runners(sess=sess)

        # Create a summary writer
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        max_steps = 201
        for step in range(max_steps):
            _, loss_value, acc = sess.run([train_op, loss, accuracy])
            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                format_str = '%s: step %d, loss = %.4f, accuracy = %.4f'
                print format_str % (datetime.now(), step, loss_value, acc)
                #with printoptions(precision=4, suppress=True):
                #    print p
                # save model to checkpoint
                if step % 100 == 0:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)