from typing import Iterable

import tensorflow as tf

from tsne_utils import tsne_loss


class ParametricTSNE:
    """
    Gabe's take on parametric tsne -- mostly RBMs become an autoencoder.
    Please make sure to train the pretrainer model first.
    """

    def __init__(self, layer_sizes: Iterable = (32, 64, 32, 2), input_size: int = 784) -> None:
        """
        Initializes the class by creating the layers used in pretraining and fine-tuning.
        :param layer_sizes: The number of layers to create, along with their sizes.
        :param input_size: The number of features of each example.
        """
        super().__init__()
        self.__input_size = input_size
        self.__encoder_layers = []
        self.__decoder_layers = []
        self.__encoder_layers.append(tf.keras.layers.Input(shape=self.__input_size))
        for layer_size in layer_sizes[0:-1]:
            self.__encoder_layers.append(
                tf.keras.layers.Dense(layer_size,
                                      activation=tf.keras.activations.relu,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_initializer=tf.constant_initializer(.0001)))
        self.__encoder_layers.append(
            tf.keras.layers.Dense(layer_sizes[-1],
                                  activation=tf.keras.activations.linear,
                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                  bias_initializer=tf.constant_initializer(.0001)))
        for layer_size in list(reversed(layer_sizes[0:-1])):
            self.__decoder_layers.append(
                tf.keras.layers.Dense(layer_size,
                                      activation=tf.keras.activations.relu,
                                      kernel_initializer=tf.keras.initializers.he_uniform(),
                                      bias_initializer=tf.constant_initializer(.0001)))
        self.__decoder_layers.append(
            tf.keras.layers.Dense(input_size,
                                  activation=tf.keras.activations.sigmoid,
                                  kernel_initializer=tf.keras.initializers.he_uniform(),
                                  bias_initializer=tf.constant_initializer(.0001)))

    def pretrainer_model(self) -> tf.keras.Model:
        """
        Get the pretrainer model. This model should be trained before actually using the tsne model.
        :return: The model to used to pretrain the layers.
        """
        inpt = self.__encoder_layers[0]
        x = inpt
        for encoder_layer in self.__encoder_layers[1:]:
            x = encoder_layer(x)
        for decoder_layer in self.__decoder_layers:
            x = decoder_layer(x)
        model = tf.keras.Model(inputs=inpt,
                               outputs=x,
                               name="parametric_tsne_pretrainer")
        model.compile(optimizer="nadam", loss='mse')
        return model

    def tsne_model(self) -> tf.keras.Model:
        """
        Gets the actual parametric tsne model, containing the pretrained layers. Make sure to
        actually pretrain the pretrainer model before using this one.
        :return: The actual parametric tsne model, containing the pretrained layers.
        """
        inpt = self.__encoder_layers[0]
        x = inpt
        for encoder_layer in self.__encoder_layers[1:]:
            x = encoder_layer(x)
        model = tf.keras.Model(inputs=inpt,
                               outputs=x,
                               name="parametric_tsne")
        model.compile(optimizer="nadam", loss=tsne_loss)
        return model
