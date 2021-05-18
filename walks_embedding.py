from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_addons as tfa
layers = tf.keras.layers

class WalksEmbedding():
    def __init__(self, params, d_model):
        if params.layer_sizes is None:
            self._layer_sizes = {'fc1': 128, 'fc2': d_model}
        else:
            self._layer_sizes = params.layer_sizes
        self._params = params
    # def _init_layers(self):
        kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        initializer = tf.initializers.Orthogonal(3)
        self._use_norm_layer = self._params.use_norm_layer is not None
        if self._params.use_norm_layer == 'InstanceNorm':
            self._norm1 = tfa.layers.InstanceNormalization(axis=2)
            self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        elif self._params.use_norm_layer == 'BatchNorm':
            self._norm1 = layers.BatchNormalization(axis=2)
            self._norm2 = layers.BatchNormalization(axis=2)
        self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)

    def run(self, model_ftrs, classify=True, skip_1st=True, training=True):
        if skip_1st:
            x = model_ftrs[:, 1:]
        else:
            x = model_ftrs
        x = self._fc1(x)
        if self._use_norm_layer:
            x = self._norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self._fc2(x)
        if self._use_norm_layer:
            x = self._norm2(x, training=training)
        x = tf.nn.relu(x)

        #  if classify:
        #      x = self._fc_last(x)

        return x
