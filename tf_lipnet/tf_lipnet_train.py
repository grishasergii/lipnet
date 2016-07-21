import tensorflow as tf
import tf_lipnet_input
import tf_lipnet
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

# Directory for train information output. It will be emptied at the beginning of every run
tf.app.flags.DEFINE_string('train_dir', './tmp/lipnet_train', """Directory where to write event logs and checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Maximum number of training epochs""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement""")


def _empty_train_dir():
    """
    Empty train directory
    :return: nothing
    """
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)


def train(particles_df, path_to_images, max_steps):
    """
    Train lipnet CNN with Tensorflow
    :param particles_df: pandas data frame that describes all particles
    :param path_to_images: path to folder with images
    :return:
    """
    _empty_train_dir()
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # batch size. pass it as parameter later
        batch_size = 100

        # get images and labels, training set
        images, labels = tf_lipnet_input.inputs(particles_df,
                                                path_to_images,
                                                batch_size=batch_size)
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.float32)
        # Build a graph that computes the logits predictions.
        # predictions - predicted probabilities of belonging to any of classes
        predictions = tf_lipnet.get_predictions(images, batch_size=batch_size)

        # calculate loss
        loss = tf_lipnet.get_loss(predictions, labels)

        tf.scalar_summary('Loss', loss)

        # Build a Graph that trains the model with one batch of examples
        # and updates the model parameters
        train_op = tf_lipnet.train(loss, global_step, batch_size)

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
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in range(1, max_steps, 1):
            _, loss_value = sess.run([train_op, loss])
            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                format_str = '%s: step %d, loss = %.2f'
                print format_str % (datetime.now(), step, loss_value)
