# -*- coding: utf8 -*-
"""
Fishnet Applications. 
Functions: 
    - fishnet99(**kwargs)
    - fishnet77(**kwargs)
    - fishnet55(**kwargs)
"""

# tensorflow
from tensorflow.keras import Model, Input

# fishnet
from .fishnet import fish


def fishnet99(**kwargs):
    """
    Fishnet 99
        - Return -> Model
            Returns a fishnet99 model. 
    """

    net_cfg = {
        #                           tail        |        body       |     head
        #                  ---------------------|-------------------|---------------
        #       stage:     input   1    2    3  |  se   3    2    1 |  1     2    3
        #                  ---------------------|-------------------|---------------
        #  input size:     [224,  56,  28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:     [ 56,  28,  14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  ---------------------|-------------------|---------------
        # num input channels  |    |    |    |     |    |    |    |    |     |    |
        'input_channels':    64, # |    |    |     |    |    |    |    |     |    |
        # num channels             |    |    |     |    |    |    |    |     |    |
        'stage_channels':      [  128, 256, 512,  512, 512, 384, 256, 320, 832, 1600  ], 
        # num residules            |    |    |     |    |    |    |    |     |    |
        'num_res_blocks':      [   2,   2,   6,    2,   1,   1,   1,   1,    2,   2   ], 
        # num transitions          |    |    |     |    |    |    |    |     |    |
        'num_trans_blocks':    [   0,   0,   0,    0,   1,   1,   1,   1,    1,   4   ], 
        # source of trans          |    |    |     |    |    |    |    |     |    |
        'trans_map':           (  -1,  -1,  -1,   -1,   2,   1,   0,   6,    5,   4   ), 
        # use concise version
        'concise': True, 
        # num stages in each part
        'num_stages': 3, 
    }

    # specify inputs
    input_shape = kwargs.get('input_shape', (56, 56, 3))
    kwargs.pop('input_shape', None)

    # specify number of classes
    num_cls = kwargs.get('num_cls', 10)
    kwargs['num_cls'] = num_cls

    # build fish
    cfg = {**net_cfg, **kwargs}
    fishnet = fish(**cfg)

    # fishnet
    inputs = Input(shape=input_shape)
    out = fishnet(inputs)

    return Model(inputs=inputs, outputs=out, name='fishnet_99')


def fishnet77(**kwargs):
    """
    Fishnet 77
        - Return -> Model
            Returns a fishnet77 model. 
    """

    net_cfg = {
        #                           tail        |        body       |     head
        #                  ---------------------|-------------------|---------------
        #       stage:     input   1    2    3  |  se   3    2    1 |  1     2    3
        #                  ---------------------|-------------------|---------------
        #  input size:     [224,  56,  28,  14  |  7,   7,  14,  28 | 56,   28,  14]
        # output size:     [ 56,  28,  14,   7  |  7,  14,  28,  56 | 28,   14,   7]
        #                  ---------------------|-------------------|---------------
        # num input channels  |    |    |    |     |    |    |    |    |     |    |
        'input_channels':    16, # |    |    |     |    |    |    |    |     |    |
        # num channels             |    |    |     |    |    |    |    |     |    |
        'stage_channels':      [  32,  64, 128,  128, 128,  96,  64,  80,  208, 400  ], 
        # num residules            |    |    |     |    |    |    |    |     |    |
        'num_res_blocks':      [   2,   2,   6,    2,   1,   1,   1,   1,   2,   2   ], 
        # num transitions          |    |    |     |    |    |    |    |     |    |
        'num_trans_blocks':    [   0,   0,   0,    0,   1,   1,   1,   1,   1,   4   ], 
        # source of trans          |    |    |     |    |    |    |    |     |    |
        'trans_map':           (  -1,  -1,  -1,   -1,   2,   1,   0,   6,   5,   4   ), 
        # use concise version
        'concise': True, 
        # num stages in each part
        'num_stages': 3, 
    }

    # specify inputs
    input_shape = kwargs.get('input_shape', (56, 56, 3))
    kwargs.pop('input_shape', None)

    num_cls = kwargs.get('num_cls', 100)
    kwargs['num_cls'] = num_cls

    # build fish
    cfg = {**net_cfg, **kwargs}
    fishnet = fish(**cfg)

    # fishnet
    inputs = Input(shape=input_shape)
    out = fishnet(inputs)

    return Model(inputs=inputs, outputs=out, name='fishnet_77')


def fishnet55(**kwargs):
    """
    Fishnet 55
        - Return -> Model
            Returns a fishnet55 model. 
    """

    net_cfg = {
        #                           tail   |        body  |     head
        #                  ----------------|--------------|------------
        #       stage:     input   1    3  |  se   3    1 |  1     3
        #                  ----------------|--------------|------------
        #  input size:     [ 56,  56,  28  |  7,   7,  14 | 28,   14]
        # output size:     [ 56,  28,  14  |  7,  14,  28 | 14,    7]
        #                  ----------------|--------------|------------
        # num input channels  |    |    |     |    |    |    |     |
        'input_channels':     8, # |    |     |    |    |    |     |
        # num channels             |    |     |    |    |    |     |
        'stage_channels':      [  16,  32,   32,  32,  24,  32,   48   ],
        # num residules            |    |     |    |    |    |     |
        'num_res_blocks':      [   2,   3,    2,   1,   1,   1,    2   ], 
        # num transitions          |    |     |    |    |    |     |
        'num_trans_blocks':    [   0,   0,    0,   1,   1,   1,    4   ], 
        # source of trans          |    |     |    |    |    |     |
        'trans_map':           (  -1,  -1,   -1,   1,   0,   4,    3   ), 
        # use concise version
        'concise': True, 
        # num stages in each part
        'num_stages': 2, 
    }

    # specify inputs
    input_shape = kwargs.get('input_shape', (56, 56, 3))
    kwargs.pop('input_shape', None)

    num_cls = kwargs.get('num_cls', 10)
    kwargs['num_cls'] = num_cls

    # build fish
    cfg = {**net_cfg, **kwargs}
    fishnet = fish(**cfg)

    # fishnet
    inputs = Input(shape=input_shape)
    out = fishnet(inputs)

    return Model(inputs=inputs, outputs=out, name='fishnet_55')