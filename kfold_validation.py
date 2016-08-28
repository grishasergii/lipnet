from kfold import KFold
from lipnet_dataset import DatasetPD, DatasetPDAugmented
from lipnet_tf import train
import lipnet_architecture as la
from lipnet_tf.model import Model
from datetime import datetime
from helpers import prepare_dir
import os
import csv


def do_validation(k, path_to_json, path_to_img, do_augmentation=False, smote_rates=None, verbose=True, epochs=30):
    batch_size = 300
    kfold = KFold(k, path_to_json, path_to_img)
    stats = [None] * k
    for i in xrange(k):
        if verbose:
            print '{}: Fold {} of {}'.format(datetime.now(), i + 1, k)
        train_df, test_df = kfold.get_datasets(i)
        if do_augmentation and smote_rates is not None:
            train_set = DatasetPDAugmented.from_dataframe(train_df, batch_size=batch_size, num_epochs=epochs,
                                                          smote_rates=smote_rates, verbose=verbose)
        else:
            train_set = DatasetPD(train_df, batch_size=batch_size, num_epochs=epochs, verbose=verbose)
        test_set = DatasetPD(test_df, batch_size=batch_size, num_epochs=1, verbose=verbose)
        model = Model(3, la.layer_definitions)
        _, validation_stats = train.train(train_set, model, test_set, verbose=True)
        stats[i] = validation_stats
    return stats


def save_stats_to_csv(stats, file_path):
    prepare_dir(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Fold', 'Loss', 'Accuracy', 'Confusion matrix'])
        for i in xrange(len(stats)):
            writer.writerow([i,
                             '{:.4f}'.format(stats[i]['loss']),
                             '{:.4f}'.format(stats[i]['acc']),
                             stats[i]['cf'].as_str])


if __name__ == '__main__':
    problems = {
        'packiging': [100, 100],
        'lamellarity': [2, 100]
    }

    for problem, smote_rates in problems.iteritems():
        print '{}: Problem {}'.format(datetime.now(), problem)
        k = 5
        path_to_json = '/home/sergii/Documents/microscopic_data/{}/particles_repaired.json'.format(problem)
        path_to_img = '/home/sergii/Documents/microscopic_data/{}/images/without_padding/'.format(problem)
        stats = do_validation(k, path_to_json, path_to_img, do_augmentation=True, smote_rates=smote_rates, epochs=30)
        save_stats_to_csv(stats, './output/stats/kfold/{}_{}.csv'.format(problem, k))
