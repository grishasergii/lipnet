from lipnet_dataset import DatasetPD
from lipnet_tf import evaluate as lptf
from lipnet_tf import FLAGS

def evaluate():
    """

    :return:
    """
    problem = 'packiging'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_test_set.json'
    path_to_img = dir + 'images/without_padding/'

    # create a dataset
    dataset = DatasetPD(path_to_json.format(problem, problem),
                        path_to_img.format(problem),
                        batch_size=500,
                        num_epochs=1)
    dataset.print_stats()
    """
    batch = dataset.next_batch()
    while batch is not None:
        print 'Batch size: {}'.format(batch.size)
        batch = dataset.next_batch()
    """
    FLAGS.batch_size = dataset.get_count()
    # start evaluation
    for _ in range(1):
        lptf.evaluate(dataset,
                          path_to_img.format(problem)
                          )

def main(argv=None):
    evaluate()


if __name__ == '__main__':
    main()