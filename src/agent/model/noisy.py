import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import get as get_activation, serialize as serialize_activation


class MyNoisyDense(Layer):
    """Own creation of a noisy dense layer."""

    def __init__(self, size, std_init=0.5, **kwargs):
        super().__init__(**kwargs)
        self.size_in = None
        self.size_out = size

        self.std_init = std_init
        self.noise_active = True

        self.mu = None
        self.sigma = None

    def build(self, input_shape):
        self.size_in = int(input_shape[-1])
        layer_shape = (self.size_out, self.size_in)

        norm = 1 / self.size_in  # Why sqrt?

        self.mu = self.add_weight(
            name="mu",
            shape=layer_shape,
            initializer=tf.keras.initializers.RandomUniform(0.0, norm),
            trainable=True,
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=layer_shape,
            initializer=tf.keras.initializers.RandomUniform(0.0, self.std_init * norm),
            trainable=True,
        )

        self.reset_noise()
        super().build(input_shape)

    def reset_noise(self):
        pass

    def set_noisy(self, active: bool):
        self.noise_active = active

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self.noise_active:
            noise = tf.random.normal((self.size_out,))
            return tf.linalg.matmul(x, self.mu, transpose_b=True) + noise
        else:
            return tf.linalg.matmul(x, self.mu, transpose_b=True)

    def get_config(self):
        return {"size": self.size_out,
                "std_init": self.std_init}


def _get_noise_vector(size):
    x = tf.random.normal((size,))
    return tf.sign(x) * tf.sqrt(tf.abs(x))


class NoisyDense(Layer):
    """Part of Noisy Nets, as used in Rainbow paper."""

    def __init__(self, size, std_init=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.size_in = None
        self.size_out = size
        self.std_init = std_init
        self.activation = get_activation(activation)
        self.noise_active = True

        self.mu = None
        self.mu_bias = None
        self.sigma = None
        self.sigma_bias = None
        self.epsilon = None
        self.epsilon_bias = None

    def build(self, input_shape):
        self.size_in = int(input_shape[-1])

        norm = 1 / np.sqrt(self.size_in)
        weight_shape = (self.size_out, self.size_in)

        self.mu = self.add_weight(
            name="mu",
            shape=weight_shape,
            initializer=tf.keras.initializers.RandomUniform(-norm, norm),
            trainable=True,
        )
        self.mu_bias = self.add_weight(
            name="mu_bias",
            shape=(self.size_out,),
            initializer=tf.keras.initializers.RandomUniform(-norm, norm),
            trainable=True,
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=weight_shape,
            initializer=tf.keras.initializers.Constant(self.std_init * norm),
            trainable=True,
        )
        self.sigma_bias = self.add_weight(
            name="sigma_bias",
            shape=(self.size_out,),
            initializer=tf.keras.initializers.Constant(self.std_init * norm),
            trainable=True,
        )
        self.epsilon = self.add_weight(
            name="epsilon",
            shape=weight_shape,
            initializer="zeros",
            trainable=False,
        )
        self.epsilon_bias = self.add_weight(
            name="epsilon_bias",
            shape=(self.size_out,),
            initializer="zeros",
            trainable=False,
        )
        self.reset_noise()
        super().build(input_shape)

    def reset_noise(self):
        epsilon_in = _get_noise_vector(self.size_in)
        epsilon_out = _get_noise_vector(self.size_out)
        self.epsilon.assign(tf.tensordot(epsilon_out, epsilon_in, axes=0))  # outer product
        self.epsilon_bias.assign(epsilon_out)

    def set_noisy(self, active: bool):
        # self.epsilon.assign(tf.zeros(self.epsilon.get_shape()))
        # self.epsilon_bias.assign(tf.zeros(self.epsilon_bias.get_shape()))
        self.noise_active = active

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self.noise_active:
            A = self.mu + self.sigma * self.epsilon
            b = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            A = self.mu
            b = self.mu_bias
        output = tf.linalg.matmul(x, A, transpose_b=True) + b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        return {"size": self.size_out,
                "std_init": self.std_init,
                "activation": serialize_activation(self.activation)}
