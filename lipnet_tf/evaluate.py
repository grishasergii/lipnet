from __future__ import division
import tensorflow as tf
from . import FLAGS
import model
import math
from datetime import datetime
import numpy as np


def evaluate(dataset):
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

        logits, predictions = model.get_predictions(images, batch_size)

        loss = model.get_loss(logits, labels)
        accuracy = model.get_accuracy(predictions, labels)

        saver = tf.train.Saver()

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_eval_dir)

        # create empty array for generated predictions
        predictions_output = None

        #evaluate_once(dataset, saver, summary_writer, loss, accuracy, summary_op, dataset.get_count(), batch_size_op, images, labels, batch_size, predictions)
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

            result_accuracy = 0
            result_loss = 0
            step = 0
            batch = dataset.next_batch()
            while batch is not None:
                FLAGS.batch_size = batch.size
                acc, l, pr = sess.run([accuracy, loss, predictions],
                                                           feed_dict={images: batch.images,
                                                                      labels: batch.labels,
                                                                      batch_size: batch.size})
                ids = np.expand_dims(batch.ids, axis=1)
                if predictions_output is not None:
                    predictions_output = np.append(predictions_output, np.append(ids, pr, axis=1), axis=0)
                else:
                    predictions_output = np.append(ids, pr, axis=1)

                result_accuracy += (acc * batch.size)
                result_loss += (l * batch.size)
                step += 1
                print "%s: evaluating batch %d of size %d" % (datetime.now(), step, batch.size)
                batch = dataset.next_batch()

            result_loss /= dataset.get_count()
            result_accuracy /= dataset.get_count()

            print "%s: accuracy = %.4f loss = %.4f examples = %d" % (datetime.now(), result_accuracy, result_loss, dataset.get_count())

            summary = tf.Summary()
            #summary.ParseFromString(sess.run(summary_op, feed_dict={images: [],
            #                                                        labels: []}))
            summary.value.add(tag='Accuracy', simple_value=result_accuracy)
            summary.value.add(tag='Loss', simple_value=result_loss)
            summary_writer.add_summary(summary, global_step)

            # sort predictions by id (first column)
            predictions_output = predictions_output[predictions_output[:, 0].argsort()]
            return predictions_output


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()