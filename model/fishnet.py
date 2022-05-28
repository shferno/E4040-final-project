# -*- coding: utf-8 -*-
"""Fishnet Module
Defines the fishnet module. 
Built on tf.keras.Model, blocks built on tf.keras.Sequential. 
Functions:
    fish(**cfg) -> Fishnet
        - Factory: Creates a fishnet instance
        - Usage: model = fish(**cfg)
Classes:
    Fish <- Model
        - Defines a Fish structure
    Fishnet <- Model
        - Defines the Fishnet
"""


# import tensorflow
import tensorflow as tf
from tensorflow.keras import Model, Input
import tensorflow.keras.layers as layers

# import building blocks
from .blocks import *


# classes
class Fish(Model):
    """
    Fish module
        - Architecture
            Input -> ? -> Output
    """

    def __init__(self, num_cls=1000, num_stages=3, input_channels=64, stage_channels=None, 
                 num_res_blocks=None, num_trans_blocks=None, trans_map=(2, 1, 0, 6, 5, 4)):
        """
        Construct Fish structure. 
            - Arguments
                num_cls: number of classes
                num_stages: number of stages in each part
                input_channels
                stage_channels
                num_res_blocks
                num_trans_blocks
                trans_map
        """

        super().__init__()

        # network parameters
        self.num_cls = num_cls
        self.num_stages = num_stages
        self.stage_channels = stage_channels
        self.depth = len(stage_channels)
        self.num_trans_blocks = num_trans_blocks
        self.num_res_blocks = num_res_blocks
        self.trans_map = trans_map

        # fish layers
        # channels (concated)
        channels = [input_channels] * (self.depth + 1)

        # store layers
        network_layers = []
        score_layers = []

        # build by stages
        for i in range(self.depth):
            # get stage params
            is_down = i not in range(self.num_stages, 2 * self.num_stages + 1)
            has_trans = i > self.num_stages
            no_sampling = i == self.num_stages

            # get stage shapes
            n_ch = self.stage_channels[i]
            trans_channels = input_channels if self.trans_map[i] == -1 else channels[self.trans_map[i] - 1]
            n_blk = self.num_res_blocks[i]
            n_trans = self.num_trans_blocks[i]

            # determin dilution & dialation
            if is_down or no_sampling:
                k = 1
                dilation = (1, 1)
            else:
                k = channels[i - 1] // n_ch
                d = 2 ** (i - self.num_stages - 1)
                dilation = (d, d)

            # add stage blocks
            network_layers.append(StgBlock(is_down, channels[i - 1], n_ch, n_blk, k, dilation, 
                                   has_trans, trans_channels, no_sampling, n_trans))

            # add score (pooling to 1*1) & se block
            if i == self.num_stages:
                score_layers.append(ScrBlock(channels[i - 1], n_ch*2))
                self.se = SeBlock(n_ch*2, n_ch)

            # calculate channels
            if i == self.num_stages - 1:
                channels[i] = n_ch * 2
            elif has_trans:
                channels[i] = n_ch + trans_channels
            else:
                channels[i] = n_ch

        # build last layer
        score_layers.append(ScrBlock(n_ch+trans_channels, self.num_cls, has_pool=True))

        self.network_layers = network_layers
        self.score_layers = score_layers


    def call(self, inputs):
        """
        Forward path, handles residules
        """

        # record states
        states = [None] * (self.depth + 1)
        states[0] = inputs

        # stages
        for i ,layer in enumerate(self.network_layers):
            #print('Stage', end=' ')
            if i < self.num_stages:
                # tail
                xin = states[i]
                states[i + 1] = layer(xin)

            elif i == self.num_stages:
                # bottom
                # layers
                score_layer = self.score_layers[0]
                se_layer = self.se

                # hidden states
                xin = states[i]
                score = score_layer(xin)
                se = se_layer(score)

                # outputs
                states[i + 1] = layer(score) * se + se

            else:
                # body & head
                # input states from layer above
                trunk_in = states[i]

                # transition states from shortcuts
                trans_in = states[self.trans_map[i]]

                # outputs
                trunk = layer(trunk_in)
                trans = layer.trans_layer(trans_in)

                # concat along channel axis
                states[i + 1] = tf.concat([trunk, trans], axis=3)

            #print(i, '->', states[i + 1].shape)

        # score output
        score_layer = self.score_layers[-1]
        score = score_layer(states[-1])

        return score


class FishNet(Model):
    """
    FishNet
        - Architechture
            Input -> cbr * 3 -> maxpool -> fish -> Output
    """

    def __init__(self, **kwargs):
        """
        Construct FishNet structure. 
        """

        super().__init__()

        # params
        input_channels = kwargs['input_channels']
        self.concise = kwargs.get('concise', False)
        kwargs.pop('concise', None)

        # conv-bn-relu blocks
        self.cbr1 = CbrBlock(input_channels // 2, strides=(2, 2))
        self.cbr2 = CbrBlock(input_channels // 2)
        self.cbr3 = CbrBlock(input_channels)

        # maxpool
        self.pool = layers.MaxPool2D(3, strides=(2, 2), padding='same')

        # build fish
        self.fish = Fish(**kwargs)


    def call(self, inputs):
        """
        Forward path
        """

        # layers
        x = inputs
        
        if not self.concise:
            x = self.cbr1(inputs)

        x = self.cbr2(x)
        x = self.cbr3(x)

        if not self.concise: 
            x = self.pool(x)

        # fish
        out = self.fish(x)

        return out


# functions
def fish(**kwargs):
    """
    Creates a FishNet instance. 
        - Arguments
            kwargs: Net configs. 
        - Return -> FishNet
            Returns a FishNet instance. 
    """

    return FishNet(**kwargs)