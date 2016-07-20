import tensorflow as tf
import tf_lipnet_input
import tf_lipnet
from datetime import datetime


def train(particles_df, path_to_images, max_steps):
    """
    Train lipnet CNN with Tensorflow
    :param particles_df: pandas data frame that describes all particles
    :param path_to_images: path to folder with images
    :return:
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
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

        #
        train_op = tf_lipnet.train(loss, global_step, batch_size)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        # start the queue runners
        tf.train.start_queue_runners(sess=sess)

        for step in range(1, max_steps, 1):
            _, loss_value = sess.run([train_op, loss])
            if step % 10 == 0:
                format_str = '%s: step %d, loss = %.2f'
                print format_str % (datetime.now(), step, loss_value)
