from lipnet_dataset import DatasetPD, DatasetPDFeatures
from lipnet_tf import train as lptf
from lipnet_tf import FLAGS
import lipnet_architecture as la

def evaluate():
    """

    :return:
    """
    problem = 'lamellarity'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'
    batch_size = 500
    FLAGS.batch_size = batch_size
    epochs = 1
    # create dataset
    dataset = DatasetPD(path_to_json.format(problem, problem, 'validation'),
                          path_to_img.format(problem),
                          batch_size=batch_size,
                          num_epochs=epochs)

    model = lptf.Model(3, la.layer_definitions)
    lptf.evaluate(dataset, model, do_restore=True)


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    main()