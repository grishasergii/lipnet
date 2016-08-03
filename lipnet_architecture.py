import collections


class LayerEnum(object):
    """
    Enumeration that represents layer types
    """
    Convolutional, FullyConnected, PoolingMax, Normalization = range(4)


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
                                         'strides,'
                                         'activation_function,'
                                         'pooling_size,'
                                         'depth_radius,'
                                         'fc_nodes,'
                                         'return_preactivations'
                                         )
LayerDefinition.__new__.__defaults__ = (None,) * len(LayerDefinition._fields)
