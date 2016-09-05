from lipnet_dataset import DatasetPD, DatasetPDFeatures, DatasetPDAugmented
from lipnet_tf import train as lptf
from lipnet_tf.model import Model
from lipnet_tf import FLAGS
import lipnet_architecture as la
from datetime import datetime
from helpers import *
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def train_on_real_images(problem_name,
                         epochs,
                         run_id=1,
                         do_validation=True,
                         do_augmentation=True,
                         do_intermediate_evaluations=False,
                         plot_weights=False,
                         early_stopping=False):
    """
    Train lipnet CNN with some framework. Currently only Tensorflow is supported
    :return:
    """
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'

    batch_size = 500
    FLAGS.batch_size = batch_size
    # create train set
    if do_augmentation:
        train_set = DatasetPDAugmented.from_json(path_to_json.format(problem_name, problem_name, 'train'),
                              path_to_img.format(problem_name),
                              batch_size=batch_size,
                              num_epochs=epochs,
                                       smote_rates=[2, 100])
    else:
        train_set = DatasetPD.from_json(path_to_json.format(problem_name, problem_name, 'train'),
                                                 path_to_img.format(problem_name),
                                                 batch_size=batch_size,
                                                 num_epochs=epochs)

    validation_set = None
    if do_validation:
        validation_set = DatasetPD.from_json(path_to_json.format(problem_name, problem_name, 'validation'),
                              path_to_img.format(problem_name),
                              batch_size=batch_size,
                              num_epochs=1)

    model = Model(3, la.layer_definitions)

    plot_path = None
    if plot_weights:
        plot_path = './output/figures/{}/conv_weights/run_{}'.format(problem_name, run_id)
        prepare_dir(plot_path, empty=True)

    train_stats, validation_stats = lptf.train(train_set,
                                               model, epochs,
                                               validation_set,
                                               verbose=True,
                                               intermediate_evaluation=do_intermediate_evaluations,
                                               plot_path=plot_path,
                                               early_stopping=early_stopping)

    return train_stats, validation_stats


def main(argv=None):
    # path
    problem_name = 'lamellarity'
    path_to_stats = 'output/train_stats/{}/{}.csv'.format(problem_name, 'train-2-200_early_stopping')
    path_to_fig = './output/figures/{}/training'.format(problem_name)

    # flags
    plot_loss_acc = False
    save_stats = True
    plot_weights = False
    interm_eval = False
    early_stopping = True
    n_runs = 1
    epochs = 100

    stats_validation = [None] * n_runs
    stats_train = [None] * n_runs

    for run_id in xrange(n_runs):
        print "{}: run {}...".format(datetime.now(), run_id + 1)
        train_stats, validation_stats = train_on_real_images(problem_name,
                                                             epochs=epochs,
                                                             run_id=run_id + 1,
                                                             do_validation=True,
                                                             do_augmentation=True,
                                                             plot_weights=plot_weights,
                                                             do_intermediate_evaluations=interm_eval,
                                                             early_stopping=early_stopping)
        stats_validation[run_id] = validation_stats
        stats_train[run_id] = train_stats

        if plot_loss_acc:
            prepare_dir(path_to_fig, empty=True)
            xticks = np.arange(1, epochs + 1, 1)
            # plot loss
            plt.clf()
            plt.plot(xticks, train_stats['loss_series'])
            plt.plot(xticks, validation_stats['loss_series'])
            axes = plt.gca()
            axes.set_ylim([0, max([max(train_stats['loss_series']), max(validation_stats['loss_series'])]) * 1.1])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(os.path.join(path_to_fig, 'loss_{}.png'.format(run_id + 1)), bbox_inches='tight')

            # plot accuracy
            plt.clf()
            plt.plot(xticks, train_stats['acc_series'])
            plt.plot(xticks, validation_stats['acc_series'])
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.xticks(xticks)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='lower right')
            plt.savefig(os.path.join(path_to_fig, 'accuracy_{}.png'.format(run_id + 1)), bbox_inches='tight')

    if save_stats:
        prepare_dir(os.path.dirname(path_to_stats))
        with open(path_to_stats, 'wb') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Run', 'Loss', 'Accuracy', 'Confusion matrix validation', 'Min loss epoch',
                             'Confusion matrix training'])
            for i in xrange(len(stats_validation)):
                writer.writerow([i + 1,
                                 '{:.4f}'.format(stats_validation[i]['loss']),
                                 '{:.4f}'.format(stats_validation[i]['acc']),
                                 stats_validation[i]['cf'].as_str,
                                '{}'.format(stats_validation[i].get('min_loss_epoch', 'n/a')),
                                '{}'.format(stats_train[i]['cf'].as_str)]
                                )


if __name__ == '__main__':
    main()
