#import keras.backend as K
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.python.ops import nn

# Modified Dropout Layer for Bayesian U-Net: also applied during testing phase
class BayesDropout(Dropout):

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

  def call(self, inputs, training=None):

    if 0. < self.rate < 1.:
        return nn.dropout(
              inputs,
              noise_shape=self._get_noise_shape(inputs),
              seed=self.seed,
              rate=self.rate)
    else:
        return inputs



