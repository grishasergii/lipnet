from __future__ import division, print_function
from . import FLAGS
import tensorflow as tf
import os.path
from evaluate import evaluate
from datetime import datetime
import plot


def prepare_dir(directory):
    """
    Prepares a directory. If it exists everything is deleted, otherwise the directory is created
    :param directory: string, path to directory to be emptied and/or created
    :return: nothing
    """
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)


def _evaluate(sess, model, dataset, verbose=True, return_confusion_matrix=False, summary_writer=None):
    return evaluate(dataset, model, session=sess, do_restore=False, verbose=verbose,
                    return_confusion_matrix=return_confusion_matrix,
                    summary_writer=summary_writer)


def train(dataset,
          model,
          epochs,
          validation_set=None,
          verbose=True,
          intermediate_evaluation=False,
          plot_path='',
          early_stopping=False):
    """

    :param dataset:
    :param model:
    :param epochs:
    :param validation_set:
    :param verbose:
    :param intermediate_evaluation:
    :param plot_path:
    :param early_stopping:
    :return:
    """
    if verbose:
        print('\r{}: Training started'.format(datetime.now()))
    global_step = 0
    total_batches = dataset.batches_count
    prepare_dir(os.path.join(FLAGS.logdir, 'train'))
    prepare_dir(os.path.join(FLAGS.logdir, 'validation'))

    #merged = tf.merge_all_summaries()

    tf.logging.set_verbosity(tf.logging.ERROR)

    train_accuracy = [0] * epochs
    train_loss = [0] * epochs
    validation_accuracy = [0] * epochs
    validation_loss = [0] * epochs

    # early stopping
    max_epochs_without_improvement = 30
    epochs_witout_improvement = 0
    min_validation_loss = 999999
    min_loss_epoch = -1

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
        validation_writer = tf.train.SummaryWriter(os.path.join(FLAGS.logdir, 'validation'), sess.graph)

        sess.run(model.init)
        epoch = 0
        while epoch < epochs and epochs_witout_improvement < max_epochs_without_improvement:
            # iterate over entire dataset in minibatches
            for i, batch in enumerate(dataset.iterate_minibatches(shuffle_=True)):
                global_step += 1
                if verbose:
                    print("\r{}: Epoch {}/{} Training batch {} of {}".format(datetime.now(),
                                                                             epoch + 1, epochs,
                                                                             i + 1, total_batches), end='')

                batch_x = batch.images
                batch_y = batch.labels

                sess.run(model.optimizer, feed_dict={model.x: batch_x,
                                                     model.y: batch_y,
                                                     model.keep_prob: model.dropout,
                                                     model.learning_rate: 0.0001})

            # evaluate training set
            if intermediate_evaluation:
                if verbose:
                    print('\r{}: Epoch {}/{} Evaluating Training set:'.format(datetime.now(),
                                                                              epoch + 1, epochs,), end='')
                    # Do evaluation on training set and write summaries
                train_loss[epoch], train_accuracy[epoch], _, train_summary = \
                    _evaluate(sess, model, dataset, verbose=False, return_confusion_matrix=False,
                              summary_writer=train_writer)
                #train_writer.add_summary(train_summary, epoch)

            save_scheckpoint = True
            current_loss = None
            # evaluate validation set
            if intermediate_evaluation or early_stopping:
                if validation_set is not None:
                    if verbose:
                        print('\r{}: Epoch {}/{} Evaluating Validation set:'.format(datetime.now(),
                                                                                    epoch + 1, epochs,), end='')
                    current_loss, validation_accuracy[epoch], _, validation_summary = \
                        _evaluate(sess, model, validation_set, verbose=False, return_confusion_matrix=False,
                                  summary_writer=validation_writer)
                    validation_loss[epoch] = current_loss
                    #validation_writer.add_summary(validation_summary, epoch)

            if early_stopping:
                if current_loss is not None:
                    if current_loss < min_validation_loss:
                        min_validation_loss = current_loss
                        save_scheckpoint = True
                        epochs_witout_improvement = 0
                        min_loss_epoch = epoch
                    else:
                        save_scheckpoint = False
                        epochs_witout_improvement += 1

            if save_scheckpoint:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=epoch)

            epoch += 1

        if validation_set is not None:
            if verbose:
                print('\r{}: Final validation set evaluation...'.format(datetime.now()), end='')
            final_loss, final_acc, final_cf, _ = _evaluate(sess, model, validation_set, verbose=verbose, return_confusion_matrix=True)

        if verbose:
            print('\r{}: Final train set evaluation...'.format(datetime.now()), end='')
        _, _, final_train_cf, _ = _evaluate(sess, model, dataset, verbose=False, return_confusion_matrix=True)

        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        model.saver.save(sess, checkpoint_path, global_step=global_step)

        if plot_path is not None:
            print('\rPlotting weights', end='')
            weights = sess.run([tf.get_collection('conv_weights')])
            for i, w in enumerate(weights[0]):
                plot.plot_conv_weights(w, 'conv{}'.format(i + 1), plot_path)

            conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={model.x: dataset.get_image_by_id(543005)})
            for i, c in enumerate(conv_out[0]):
                plot.plot_conv_output(c, 'conv{}'.format(i + 1), plot_path)

    if verbose:
        print('\r{}: Training finished'.format(datetime.now()))

    output = {
        #"final_batch_loss": "%.4f" % loss_batch,
        #"final_batch_acc": "%.4f" % batch_acc,
    }
    train_stats = {}
    validation_stats = None
    if validation_set is not None:
        validation_stats = {}
        validation_stats['loss'] = final_loss
        validation_stats['acc'] = final_acc
        validation_stats['cf'] = final_cf
        if intermediate_evaluation:
            validation_stats['loss_series'] = validation_loss
            validation_stats['acc_series'] = validation_accuracy
            train_stats['loss_series'] = train_loss
            train_stats['acc_series'] = train_accuracy
        if early_stopping:
            validation_stats['min_loss_epoch'] = min_loss_epoch

    train_stats['cf'] = final_train_cf

    """
    if validation_set is not None and validation_stats is not None:
        min_validation_loss_i = validation_stats.argmin(axis=0)[0]
        max_validation_acc_i = validation_stats.argmax(axis=0)[1]
        output["z_confusion_matrix_final"] = ["%.2f" % x for x in validation_cf[-1, :]],
        output["min_validation_loss:"] = "%.4f" % validation_stats[min_validation_loss_i, 0],
        output["max_validation_acc"] = "%.4f" % validation_stats[max_validation_acc_i, 1],
        output["min_validation_loss_epoch"] = dataset.step_to_epoch((min_validation_loss_i + 1) * eval_step),
        output["z_confusion_matrix_min_loss"] = ["%.2f" % x for x in validation_cf[min_validation_loss_i, :]],
        output["max_validation_acc_epoch"] = dataset.step_to_epoch((max_validation_acc_i + 1) * eval_step),
        output["z_confusion_matrix_max_acc"] = ["%.2f" % x for x in validation_cf[max_validation_acc_i, :]]
    """
    return train_stats, validation_stats