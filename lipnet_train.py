from lipnet_dataset import DatasetPD, DatasetPDFeatures, DatasetPDAugmented
from lipnet_tf import train as lptf
from lipnet_tf import FLAGS
from lipnet_architecture import *


def train_on_images():
    """
    Train lipnet CNN with some framework. Currently only Tensorflow is supported
    :return:
    """
    problem = 'lamellarity'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'
    batch_size = 10
    FLAGS.batch_size = batch_size
    epochs = 1
    # create train set
    train_set = DatasetPD(path_to_json.format(problem, problem, 'train'),
                          path_to_img.format(problem),
                          batch_size=batch_size,
                          num_epochs=epochs)

    validation_set = DatasetPD(path_to_json.format(problem, problem, 'validation'),
                               path_to_img.format(problem),
                               batch_size=None,
                               num_epochs=1)

    # define network architecture
    layer_definitions = [
        LayerDefinition(layer_type=LayerEnum.Convolutional,
                        name='conv1',
                        filter_size=[3, 3],
                        filter_num=64,
                        strides=[1, 1],
                        activation_function=ActivationFunctionEnum.Relu),

        LayerDefinition(layer_type=LayerEnum.PoolingMax,
                        name='pooling1',
                        pooling_size=[3, 3],
                        strides=[2, 2]),

        LayerDefinition(layer_type=LayerEnum.Normalization,
                        name='norm1',
                        depth_radius=5),

        LayerDefinition(layer_type=LayerEnum.Convolutional,
                        name='conv2',
                        filter_size=[3, 3],
                        filter_num=32,
                        strides=[1, 1],
                        activation_function=ActivationFunctionEnum.Relu),

        LayerDefinition(layer_type=LayerEnum.PoolingMax,
                        name='pooling2',
                        pooling_size=[3, 3],
                        strides=[2, 2]),

        LayerDefinition(layer_type=LayerEnum.Normalization,
                        name='norm2',
                        depth_radius=5),

        LayerDefinition(layer_type=LayerEnum.FullyConnected,
                        name='fc1',
                        fc_nodes=384,
                        activation_function=ActivationFunctionEnum.Relu,
                        return_preactivations=False),

        LayerDefinition(layer_type=LayerEnum.FullyConnected,
                        name='fc2',
                        fc_nodes=192,
                        activation_function=ActivationFunctionEnum.Relu,
                        return_preactivations=False),

        LayerDefinition(layer_type=LayerEnum.FullyConnected,
                        name='softmax_linear',
                        fc_nodes=train_set.get_num_classes(),
                        activation_function=ActivationFunctionEnum.Softmax,
                        return_preactivations=True),

    ]

    # start training
    """
    lptf.train(train_set,
                None,
                layer_definitions)
    """
    lptf.train_simple(train_set,
                      validation_set)


def train_on_features():
    problem = 'lamellarity'
    dir = '/home/sergii/Documents/microscopic_data/{}/'
    path_to_json = dir + '{}_{}_set.json'
    path_to_img = dir + 'images/without_padding/'
    batch_size = 500
    FLAGS.batch_size = batch_size
    epochs = 1000
    # create train set
    train_set = DatasetPDFeatures(path_to_json.format(problem, problem, 'train'),
                          path_to_img.format(problem),
                          batch_size=batch_size,
                          num_epochs=epochs)

    layer_definitions = [
        LayerDefinition(layer_type=LayerEnum.FullyConnected,
                        name='fc1',
                        fc_nodes=20,
                        activation_function=ActivationFunctionEnum.Relu,
                        return_preactivations=False),

        LayerDefinition(layer_type=LayerEnum.FullyConnected,
                        name='softmax_linear',
                        fc_nodes=train_set.get_num_classes(),
                        activation_function=ActivationFunctionEnum.Softmax,
                        return_preactivations=True),
    ]
    # start training
    lptf.train(train_set,
               None,
               layer_definitions)


def main(argv=None):
    #train_on_features()
    train_on_images()

if __name__ == '__main__':
    main()
