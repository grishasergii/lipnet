from __future__ import division
import tensorflow as tf
from . import FLAGS
import tf_lipnet
import math
from datetime import datetime


def evaluate_once(dataset, saver, summary_writer, loss_op, accuracy_op, summary_op, num_examples, batch_size_op,
                  images_ph,
                  labels_ph,
                  batch_size_ph):
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
            print 'Restoring from {}'.format(checkpoint.model_checkpoint_path)
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
            total_sample_count = 0
            step = 0
            batch = dataset.next_batch()
            while batch is not None:
                FLAGS.batch_size = batch.size
                acc, l, batch_size = sess.run([accuracy_op, loss_op, batch_size_op], feed_dict={images_ph: batch.images,
                                                                                                labels_ph: batch.labels,
                                                                                                batch_size_ph: batch.size})
                accuracy += (acc * batch_size)
                loss += (l * batch_size)
                total_sample_count += batch_size
                step += 1
                print "%s: evaluating batch %d of size %d" % (datetime.now(), step, batch_size)
                batch = dataset.next_batch()

            loss /= total_sample_count
            accuracy /= total_sample_count

            print "%s: accuracy = %.4f loss = %.4f examples = %d" % (datetime.now(), accuracy, loss, total_sample_count)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op, feed_dict={images_ph: [],
                                                                    labels_ph: [],
                                                                    batch_size_ph: 0}))
            summary.value.add(tag='Accuracy', simple_value=accuracy)
            summary.value.add(tag='Loss', simple_value=loss)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset, path_to_images):
    """

    :param dataset:
    :param path_to_images:
    :param batch_size:
    :return:
    """
    if tf.gfile.Exists(FLAGS.log_eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_eval_dir)
    tf.gfile.MakeDirs(FLAGS.log_eval_dir)

    with tf.Graph().as_default() as g:

        images = tf.placeholder(tf.float32, [None, FLAGS.image_width, FLAGS.image_height, 1], name='images_input')
        labels = tf.placeholder(tf.float32, [None, dataset.get_num_classes()], name='labels_input')
        batch_size = tf.placeholder(tf.int32, name='batch_size')

        logits, predictions = tf_lipnet.get_predictions(images, batch_size)

        loss = tf_lipnet.get_loss(logits, labels)
        accuracy = tf_lipnet.get_accuracy(predictions, labels)
        batch_size_op = tf_lipnet.get_batch_size(images, labels)

        saver = tf.train.Saver()

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_eval_dir)

        evaluate_once(dataset, saver, summary_writer, loss, accuracy, summary_op, dataset.get_count(), batch_size_op, images, labels, batch_size)



def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()