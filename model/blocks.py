# -*- coding: utf8 -*-
"""
FishNet Blocks. 
Defines
Classes:
    BrcBlock
        - Defines the building blocks of FishNet
"""


# import tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers


# classes
class BrcBlock(Sequential):
    """
    BatchNormalization-ReLU-Convolution Block
        - Architechture
            bn -> relu -> conv
    """

    def __init__(self, output_channels, kernel_size, dilation=(1, 1)):
        """
        Creates the Block. 
            - Arguments
                output_channels
                stride
        """

        super().__init__()

        # add layers
        self.add(layers.BatchNormalization())
        self.add(layers.ReLU())
        self.add(layers.Conv2D(output_channels, kernel_size, padding='same', 
                               dilation_rate=dilation, use_bias=False))


class Bottleneck(Model):
    """
    Bottleneck Block
        - Architechture
            Input -> brc * 3 -> Output + Input
    """

    def __init__(self, input_channels, output_channels, mode='NORM', k=1, dilation=(1, 1)):
        """
        Creates the Bottleneck Block
            - Arguments
                input_channels
                output_channels
                mode: [ 'NORM' | 'UP' ], indicates down sampling or up sampling
                k: channel reduction ratio
                dilation
        """

        super().__init__()

        self.mode = mode
        self.k = k
        neck_channels = output_channels // 4

        self.brc1 = BrcBlock(neck_channels, 1)
        self.brc2 = BrcBlock(neck_channels, 3, dilation=dilation)
        self.brc3 = BrcBlock(output_channels, 1)

        self.ic = input_channels
        self.oc = output_channels

        if mode == 'UP':
            # no shorcuts in up-sampling
            self.shortcut = None
        elif input_channels != output_channels:
            # use shortcut when channels differ
            self.shortcut = BrcBlock(output_channels, 1)
        else:
            self.shortcut = None


    def squeeze(self, x):
        """
        Squeeze residule into output shape
        """

        _, h, w, c = x.get_shape()
        out = layers.Reshape((h, w, c // self.k, self.k))(x)
        return tf.math.reduce_sum(out, axis=-1)


    def call(self, x):
        """
        Forward path
        """

        residule = x

        out = self.brc1(x)
        out = self.brc2(out)
        out = self.brc3(out)

        if self.mode == 'UP':
            residule = self.squeeze(x)
        elif self.shortcut is not None:
            residule = self.shortcut(residule)

        out += residule

        return out


class CbrBlock(Sequential):
    """
    Convolution-BatchNormalization-ReLU Block
        - Architechture
            Input -> conv -> bn -> relu -> Output
    """

    def __init__(self, output_channels, strides=(1, 1)):
        """
        Creates the CbrBlock. 
            - Arguments
                output_channels
                stride
        """

        super().__init__()

        self.add(layers.Conv2D(output_channels, 3, padding='same', strides=strides, use_bias=False))
        self.add(layers.BatchNormalization())
        self.add(layers.ReLU())


class ResBlock(Model):
    """
    Residule Block. 
        - Architechture
            Input -> bottle * n -> Output
    """

    def __init__(self, input_channels, output_channels, nstage, is_up=False, k=1, dilation=(1, 1)):
        """
        Creates the CbrBlock. 
            - Arguments
                output_channels
                nstage: number of transition blocks
                is_up: indicates up or down sampling
                k: channel reduction ratio
                dilation
        """

        super().__init__()

        # store all layers
        network_layers = []

        # add 1st layer
        if is_up:
            network_layers.append(Bottleneck(input_channels, output_channels, mode='UP', k=k, dilation=dilation))
        else:
            network_layers.append(Bottleneck(input_channels, output_channels))

        # add other layers
        for _ in range(1, nstage):
            network_layers.append(Bottleneck(output_channels, output_channels, dilation=dilation))

        self.network_layers = network_layers


    def call(self, inputs):
        """
        Forward path
        """

        x = inputs
        for layer in self.network_layers:
            x = layer(x)

        return x


class StgBlock(Model):
    """
    Stage Block. 
        - Architechture
            Input -> res -> [res(trans)] -> [sample] -> Output
    """

    def __init__(self, is_down_sample, input_channels, output_channels, nstage, k, dilation, 
                 has_trans=True, trans_channels=0, no_sampling=False, num_trans=2):
        """
        Creates the CbrBlock. 
            - Arguments
                is_down_sample: indicates down or up sampling
                inpput_channels
                output_channels
                n_blk: numbeer of residule blocks
                k: channel reduction ratio
                dilation
                has_trans: indicates if transition inputs exist
                has_score: indicates if calculating outputs
                trans_channels: number of channels in transition inputs
                no_sampling: indicates if sampling operation should be used
                num_trans: number of transition inupts
        """

        super().__init__()

        # store layers
        network_layers = []

        # add res layer
        if no_sampling or is_down_sample:
            network_layers.append(ResBlock(input_channels, output_channels, nstage, k=k, dilation=dilation))
        else:
            network_layers.append(ResBlock(input_channels, output_channels, nstage, is_up=True, k=k, dilation=dilation))

        # add trans layer
        if has_trans:
            self.trans_layer = ResBlock(trans_channels, trans_channels, num_trans)
        else:
            self.trans_layer = None

        # add sampling layer
        if not no_sampling and is_down_sample:
            network_layers.append(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)))
        elif not  no_sampling:
            network_layers.append(tf.keras.layers.UpSampling2D((2, 2)))

        self.network_layers = network_layers


    def call(self, inputs):
        """
        Forward path
        """

        x = inputs
        for layer in self.network_layers:
            x = layer(x)

        return x


class ScrBlock(Sequential):
    """
    Score block. 
        - Architechture
            Input 
            -> bn -> relu -> conv -> bn
            -> [avgpool] -> conv
            -> Output
    """

    def __init__(self, input_channels, output_channels=1000, has_pool=False):
        """
        Creates the CbrBlock. 
            - Arguments
                input_channels
                output_channels
                has_pool: indicates if pooling (flatten) is used
        """

        super().__init__()

        # add conv layers
        self.add(layers.BatchNormalization())
        self.add(layers.ReLU())
        self.add(layers.Conv2D(input_channels // 2, 1, use_bias=False))
        self.add(layers.BatchNormalization())

        # pooling
        if has_pool:
            self.add(layers.Flatten())
            self.add(layers.Dense(output_channels, activation='softmax'))
            #self.add(layers.GlobalAveragePooling2D())
            #self.add(layers.Reshape((1, 1, input_channels // 2)))
        else:
            # add conv
            self.add(layers.Conv2D(output_channels, 1))


class SeBlock(Sequential):
    """
    Squeeze and Excitation Block. 
        - Architechture
            Input -> bn -> relu -> avgpool 
            -> conv(s) -> relu 
            -> conv(e) -> sigmoid
            -> Output
    """

    def __init__(self, input_channels, output_channels):
        """
        Creates the CbrBlock. 
            - Arguments
                output_channels
        """

        super().__init__()

        # add layers
        self.add(layers.BatchNormalization())
        self.add(layers.ReLU())
        self.add(layers.GlobalAveragePooling2D())
        self.add(layers.Reshape((1, 1, input_channels)))

        # add squeeze
        self.add(layers.Conv2D(output_channels // 16, 1, activation='relu'))

        # add excitation
        self.add(layers.Conv2D(output_channels, 1, activation='sigmoid'))