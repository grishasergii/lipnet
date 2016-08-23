import collections


class LayerEnum(object):
    """
    Enumeration that represents layer types
    """
    Convolutional, FullyConnected, PoolingMax, Normalization, Output = range(5)


class ActivationFunctionEnum(object):
    """
    Enumeration that represents types of activation functions
    """
    Relu, Sigmoid, Softmax = range(3)

# Named tuple holding parameters of all supported layer types
LayerDefinition = collections.namedtuple('LayerDefinition',
                                         'layer_type,'
                                         'name,'
                                         'filter_size,'
                                         'filter_num,'
                                         'stride,'
                                         'activation_function,'
                                         'pooling_size,'
                                         'depth_radius,'
                                         'fc_nodes,'
                                         'return_preactivations,'
                                         'keep_prob'
                                         )
LayerDefinition.__new__.__defaults__ = (None,) * len(LayerDefinition._fields)

layer_definitions = [
    LayerDefinition(layer_type=LayerEnum.Convolutional,
                    name='conv1',
                    filter_size=[4, 4],
                    filter_num=32,
                    stride=1,
                    activation_function=ActivationFunctionEnum.Relu),

    LayerDefinition(layer_type=LayerEnum.PoolingMax,
                    name='pooling1',
                    pooling_size=2,
                    stride=2),

    LayerDefinition(layer_type=LayerEnum.Convolutional,
                    name='conv2',
                    filter_size=[4, 4],
                    filter_num=64,
                    stride=1,
                    activation_function=ActivationFunctionEnum.Relu),

    LayerDefinition(layer_type=LayerEnum.PoolingMax,
                    name='pooling3',
                    pooling_size=2,
                    stride=2),

    LayerDefinition(layer_type=LayerEnum.FullyConnected,
                    name='fc1',
                    fc_nodes=1024,
                    activation_function=ActivationFunctionEnum.Relu,
                    return_preactivations=False),

    LayerDefinition(layer_type=LayerEnum.Output,
                    name='output',
                    fc_nodes=3,
                    activation_function=ActivationFunctionEnum.Softmax,
                    return_preactivations=True),
]
