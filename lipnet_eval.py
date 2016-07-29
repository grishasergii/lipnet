from lipnet_dataset import DatasetPD
from lipnet_tf import evaluate as lptf
from lipnet_tf import FLAGS
import numpy as np
import os
from datetime import datetime
import confusion_matrix as cf

problem = 'packiging'
dir = '/home/sergii/Documents/microscopic_data/{}/'
path_to_json = dir + '{}_test_set.json'
path_to_img = dir + 'images/without_padding/'

def evaluate():
    """

    :return:
    """
    # create a dataset
    dataset = DatasetPD(path_to_json.format(problem, problem),
                        path_to_img.format(problem),
                        batch_size=500,
                        num_epochs=1)
    dataset.print_stats()

    FLAGS.batch_size = dataset.get_count()
    # start evaluation

    predictions = lptf.evaluate(dataset)

    output_path = 'output/predictions/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    output_path += 'predictions.csv'
    np.savetxt(output_path, predictions, fmt='%06d, %1.4f, %1.4f, %1.4f')
    confusion_matrix = cf.ConfusionMatrix(predictions[:, 1:], dataset.get_id_sorted_labels())
    confusion_matrix.print_to_console()

def analyze():
    # create a dataset
    dataset = DatasetPD(path_to_json.format(problem, problem),
                        path_to_img.format(problem),
                        batch_size=500,
                        num_epochs=1)
    predictions = np.loadtxt('output/predictions/predictions.csv', delimiter=',')
    confusion_matrix = cf.ConfusionMatrix(predictions[:, 1:], dataset.get_id_sorted_labels())
    confusion_matrix.print_to_console()


def main(argv=None):
    #evaluate()
    analyze()

if __name__ == '__main__':
    main()