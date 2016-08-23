from lipnet_dataset import DatasetPD, DatasetPDFeatures, DatasetPDAugmented
from lipnet_tf import train as lptf
from lipnet_tf.model import Model
from lipnet_tf import FLAGS
import lipnet_architecture as la
from datetime import datetime
from helpers import *
import csv


def train_on_images(problem_name, epochs, n_runs=1, do_validation=True, path_to_stats=None):
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
    train_set = DatasetPDAugmented(path_to_json.format(problem_name, problem_name, 'train'),
                          path_to_img.format(problem_name),
                          batch_size=batch_size,
                          num_epochs=epochs,
                                   smote_rates=[2, 200])

    validation_set = None
    if do_validation:
        validation_set = DatasetPD(path_to_json.format(problem_name, problem_name, 'validation'),
                              path_to_img.format(problem_name),
                              batch_size=batch_size,
                              num_epochs=1)

    model = Model(3, la.layer_definitions)

    stats = [None] * n_runs
    for i in xrange(n_runs):
        train_set.reset()
        validation_set.reset()
        print "{}: run {}...".format(datetime.now(), i+1)
        train_stats = lptf.train(train_set, model, validation_set, verbose=False)
        print train_stats
        stats[i] = train_stats

    if path_to_stats is not None:
        prepare_dir(os.path.dirname(path_to_stats))
        with open(path_to_stats, 'wb') as f:
            writer = csv.DictWriter(f, stats[0].keys(), delimiter=';')
            writer.writeheader()
            writer.writerows(stats)


def main(argv=None):
    problem_name = 'lamellarity'
    path_to_stats = 'output/train_stats/{}/{}.csv'.format(problem_name, 'train-2-200')
    train_on_images(problem_name, epochs=50, n_runs=10, path_to_stats=path_to_stats)

if __name__ == '__main__':
    main()
