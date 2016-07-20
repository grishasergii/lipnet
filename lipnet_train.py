import lipnet_input
from tf_lipnet import tf_lipnet_train

def train():
    """
    Train lipnet CNN with some framework. Currently only Tensorflow is supported
    :return:
    """
    dir = '/home/sergii/Documents/microscopic_data/packiging/'
    path_to_json = dir + 'particles_repaired_2.json'
    path_to_img = dir + 'images/particles/'
    df = lipnet_input.get_particles_df(path_to_json)
    tf_lipnet_train.train(df, path_to_img, 100000)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
