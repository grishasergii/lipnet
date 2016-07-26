from __future__ import division
import tensorflow as tf
from . import FLAGS
import tf_lipnet_input
import tf_lipnet
import math
from datetime import datetime


def evaluate_once(saver, summary_writer, loss_op, accuracy_op, summary_op, num_examples):
    """

    :param saver:
    :param summary_writer:
    :param loss_op:
    :param accuracy_op:
    :param summary_op:
    :return:
    """
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # Restore from checkpoint
            print 'Restoring from {}' (checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /output/model.ckpt-0,
            # extract global_step from it.
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print 'No checkpoint file found'
            return

        # Start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples/FLAGS.batch_size))
            accuracy = 0
            loss = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                acc, l = sess.run([accuracy_op, loss_op])
                accuracy += acc
                loss += l
                step += 1

            loss /= num_iter
            accuracy /= num_iter

            print "%s: accuracy = %.4f loss = %.4f" % (datetime.now(), accuracy, loss)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Accuracy', simple_value=accuracy)
            summary.value.add(tag='Loss', simple_value=loss)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(particles_df, path_to_images):
    """

    :param particles_df:
    :param path_to_images:
    :param batch_size:
    :return:
    """
    with tf.Graph().as_default() as g:
        images, labels = tf_lipnet_input.inputs(particles_df, path_to_images)

        logits, predictions = tf_lipnet.get_predictions(images, FLAGS.batch_size)

        loss = tf_lipnet.get_loss(logits, labels)
        accuracy = tf_lipnet.get_accuracy(predictions, labels)

        saver = tf.train.Saver()

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_eval_dir)

        evaluate_once(saver, summary_writer, loss, accuracy, summary_op)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.log_eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_eval_dir)
    tf.gfile.MakeDirs(FLAGS.log_eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()