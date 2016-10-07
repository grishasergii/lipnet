from __future__ import division
from kfold import KFold
from datetime import datetime
from helpers import prepare_dir
import os
import csv
import svm
from dataset.dataset import DatasetVironovaSVM
from dataset.dataset_images import DatasetImages
import numpy as np
from lipnet_keras.model_cnn import ModelLipnet4, ModelLipnet6


path_to_json = '/home/sergii/Documents/microscopic_data/{}/particles_repaired.json'
path_to_img_with_padding = '/home/sergii/Documents/microscopic_data/{}/images/particles/'
path_to_img_without_padding = '/home/sergii/Documents/microscopic_data/{}/images/without_padding/'
stats_path = 'output/stats/kfold/{}/{}.csv'
problems = [
    ('packiging', ['Empty', 'Full', 'Uncertain'])#,
    #('lamellarity', ['Unilamellar', 'Multilamellar', 'Uncertain'])
]


def cnn_fold(k, path_to_json, path_to_img, epochs=10, img_size=(28, 28), verbose=False):
    kfold = KFold(k, path_to_json, path_to_img)
    stats = [None] * 5
    for i in xrange(k):
        print '{}: Fold {} of {}'.format(datetime.now(), i + 1, k)
        train_df, test_df = kfold.get_datasets(i)
        train_set = DatasetImages(train_df, img_size)
        train_set.oversample()
        test_set = DatasetImages(test_df, img_size)
        model = ModelLipnet4(verbose=True)
        model.fit(train_set=train_set,
                  test_set=None,
                  nb_epoch=epochs)
        stats[i] = model.evaluate(test_set)
    return stats


def svm_folds(k, path_to_json):
    kfold = KFold(k, path_to_json, '')
    stats = [None] * k
    for i in xrange(k):
        print '{}: Fold {} of {}'.format(datetime.now(), i + 1, k)
        # get train and test dataframes
        train_df, test_df = kfold.get_datasets(i)

        # create train and test datasets
        test_set = DatasetVironovaSVM(train_df, do_oversampling=False)
        train_set = DatasetVironovaSVM(train_df, do_oversampling=False)

        # get confusion matrix from SVM model
        cf = svm.svm(train_set, test_set)
        stats[i] = cf
    return stats


def save_stats(cf_matrices, out_path):
    prepare_dir(os.path.dirname(out_path))
    with open(out_path, 'wb') as f:
        # create csv writer
        writer = csv.writer(f, delimiter=';')

        # write header row
        writer.writerow(['Fold', 'Sensitivity', 'Specificity', 'Precision', 'negative_predictive_value'])

        averages = np.zeros(3 * 8)
        # write a row
        for i, cfm in enumerate(cf_matrices):
            writer.writerow([i+1])
            measures_avg = []
            for ct in cfm.confusion_tables:
                row = []
                measures = [ct.true_positive,
                            ct.false_negative,
                            ct.false_positive,
                            ct.true_negative,
                            ct.sensitivity,
                            ct.specificity,
                            ct.precision,
                            ct.negative_predictive_value
                            ]
                row += [ct.name]
                for m in measures:
                    row += ['{:.4f}'.format(m)]
                measures_avg += measures
                writer.writerow(row)

            writer.writerow([cfm.as_str])

            averages = np.add(averages, measures_avg)

        k = len(cf_matrices)
        averages = np.multiply(averages, 1.0 / k)
        row = ['average']
        writer.writerow(row)
        row = []
        for x in averages:
            row += ['{:.4f}'.format(x)]
        writer.writerow(row)


def validate_cnn(model_name, path_img, img_size):
    k = 5
    verbose = False
    epochs = [70]

    for n_epochs in epochs:
        print '{}: epochs: {}'.format(datetime.now(), n_epochs)
        for problem in problems:
            print '{}: problem: {}'.format(datetime.now(), problem[0])
            stats = cnn_fold(k,
                             path_to_json.format(problem[0]),
                             path_img.format(problem[0]),
                             epochs=n_epochs,
                             verbose=verbose,
                             img_size=img_size)
            for i, _ in enumerate(stats):
                stats[i].class_names = problem[1]
            save_stats(stats, stats_path.format(model_name, problem[0]))


def validate_svm():
    k = 5
    for problem in problems:
        stats = svm_folds(k, path_to_json.format(problem[0]))
        save_stats(stats, stats_path.format('svm', problem[0]))


if __name__ == '__main__':
    #validate_svm()
    validate_cnn(model_name='lipnet4_without_surrounding_all_transformations',
                 path_img=path_to_img_without_padding,
                 img_size=(28, 28))