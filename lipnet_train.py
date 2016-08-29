from lipnet_dataset import DatasetPD, DatasetPDFeatures, DatasetPDAugmented
from lipnet_tf import train as lptf
from lipnet_tf.model import Model
from lipnet_tf import FLAGS
import lipnet_architecture as la
from datetime import datetime
from helpers import *
import csv
import pandas as pd


def train_on_real_images(problem_name, epochs, n_runs=1, do_validation=True, path_to_stats=None):
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
                                   smote_rates=[200, 200])

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
        if validation_set is not None:
            validation_set.reset()
        print "{}: run {}...".format(datetime.now(), i+1)
        train_stats = lptf.train(train_set, model, validation_set, verbose=True)
        print train_stats
        stats[i] = train_stats

    if path_to_stats is not None:
        prepare_dir(os.path.dirname(path_to_stats))
        with open(path_to_stats, 'wb') as f:
            writer = csv.DictWriter(f, stats[0].keys(), delimiter=';')
            writer.writeheader()
            writer.writerows(stats)


def train_on_synthetic_images(path_to_json, path_to_img, epochs=50):
    train_set = DatasetPD(path_to_json, path_to_img, batch_size=500, num_epochs=epochs)
    model = Model(3, la.layer_definitions)
    train_set.reset()
    train_stats = lptf.train(train_set, model)


def train_on_mixed_set(path, epochs):
    """

    :param path: list of dictionarier with keys 'path_to_json' and 'path_to_img'
    :param epochs: int
    :return:
    """
    data_set = DatasetPDAugmented(path[0]['path_to_json'], path[0]['path_to_img'],
                                  batch_size=500, num_epochs=epochs, smote_rates=[2, 0])
    for i in xrange(1, len(path)):
        df = pd.read_json(path[i]['path_to_json'])
        df['Image'] = path[i]['path_to_img'] + df['Image'].astype(str)
        data_set.add_dataframe(df)
    data_set.reset()
    model = Model(3, la.layer_definitions)
    train_stats = lptf.train(data_set, model)


def main(argv=None):
    problem_name = 'packiging'
    path_to_stats = 'output/train_stats/{}/{}.csv'.format(problem_name, 'train-2-200')
    train_on_real_images(problem_name, epochs=20, n_runs=1, path_to_stats=path_to_stats, do_validation=False)
    #train_on_synthetic_images(path_to_json='./output/synthetic_examples/synthetic_particles.json',
    #                          path_to_img='./output/synthetic_examples/synthetic_images',
    #                          epochs=20)
    path = [
        {
            'path_to_json': '/home/sergii/Documents/microscopic_data/lamellarity/lamellarity_train_set.json',
            'path_to_img': '/home/sergii/Documents/microscopic_data/lamellarity/images/without_padding/'
        },
        {
            'path_to_json': './output/synthetic_examples/lamellarity_uncertain/uncertain_particles.json',
            'path_to_img': './output/synthetic_examples/lamellarity_uncertain/'
        }
    ]
    #train_on_mixed_set(path, epochs=20)



if __name__ == '__main__':
    main()
