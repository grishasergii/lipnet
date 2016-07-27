from lipnet_dataset import DatasetPD
from tf_lipnet import tf_lipnet_eval
from tf_lipnet import FLAGS

def evaluate():
    """

    :return:
    """
    problem = 'packiging'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_test_set.json'
    path_to_img = dir + 'images/without_padding/'

    # create a dataset
    dataset = DatasetPD(path_to_json.format(problem, problem))
    dataset.print_stats()
    FLAGS.batch_size = dataset.get_count()
    # start evaluation
    for _ in range(10):
        tf_lipnet_eval.evaluate(dataset,
                                path_to_img.format(problem)
                                )


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    main()