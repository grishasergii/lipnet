from lipnet_dataset import DatasetPD
from tf_lipnet import tf_lipnet_train
from tf_lipnet import FLAGS

def train():
    """
    Train lipnet CNN with some framework. Currently only Tensorflow is supported
    :return:
    """
    problem = 'packiging'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'
    batch_size = 500
    num_epochs = 10
    FLAGS.batch_size = batch_size
    # create train set
    train_set = DatasetPD(path_to_json.format(problem, problem, 'train'),
                          path_to_img.format(problem),
                          batch_size=batch_size,
                          num_epochs=1)

    validation_set = DatasetPD(path_to_json.format(problem, problem, 'validation'),
                               path_to_img.format(problem),
                               batch_size=batch_size,
                               num_epochs=1)
    #train_set.print_stats()
    # start training
    tf_lipnet_train.train(train_set,
                          None,
                          path_to_img.format(problem),
                          10000)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
