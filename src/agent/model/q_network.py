from abc import ABCMeta

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Layer

from src.agent.model.noisy import NoisyDense


class QNetwork(Layer, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(QNetwork, self).__init__(**kwargs)
        self.num_actions = None
        self.sequential = None
        self.q_value_axis = None

    def set_num_actions(self, num_actions):
        self.num_actions = num_actions

    def set_sequential(self, sequential):
        self.sequential = sequential
        if sequential:
            self.q_value_axis = 2
        else:
            self.q_value_axis = 1

    def check_initialization(self):
        if self.num_actions is None:
            raise ValueError("You must specify number of actions first before building the Q-network.")
        elif self.sequential is None:
            raise ValueError("You must set (non-)sequential before building the Q-network.")

    def reset_noise(self):
        pass

    def set_noisy(self, active):
        pass


class DoubleQNetwork(QNetwork):
    def __init__(self, latent_v_dim=None, latent_a_dim=None, noise_std_init=0, activation=None):
        super().__init__(name="double_Q_network")
        self.v_h_size = latent_v_dim
        self.a_h_size = latent_a_dim
        self.noise_std_init = noise_std_init
        self.activation = activation
        self.v_h = None
        self.v = None
        self.a_h = None
        self.a = None

    def build(self, input_shape=None):
        self.check_initialization()

        # State value V
        if self.v_h_size is not None:
            if self.noise_std_init == 0:
                self.v_h = Dense(self.v_h_size, name='latent_V', activation=self.activation)
            else:
                self.v_h = NoisyDense(self.v_h_size, std_init=self.noise_std_init,
                                      name='latent_V', activation=self.activation)
        else:
            self.v_h = Layer()

        if self.noise_std_init == 0:
            self.v = Dense(1, name='V')  # , kernel_initializer=VarianceScaling(scale=2))
        else:
            self.v = NoisyDense(1, self.noise_std_init, name='V')

        # Advantage A
        if self.a_h_size is not None:
            if self.noise_std_init == 0:
                self.a_h = Dense(self.a_h_size, name='latent_A', activation=self.activation)
            else:
                self.a_h = NoisyDense(self.a_h_size, std_init=self.noise_std_init,
                                      name='latent_A', activation=self.activation)
        else:
            self.a_h = Layer()

        if self.noise_std_init == 0:
            self.a = Dense(self.num_actions, name='A')  # , kernel_initializer=VarianceScaling(scale=2))
        else:
            self.a = NoisyDense(self.num_actions, self.noise_std_init, name='A')

        super(DoubleQNetwork, self).build(input_shape)

    def get_config(self):
        config = {"latent_v_dim": self.v_h_size,
                  "latent_a_dim": self.a_h_size,
                  "noise_std_init": self.noise_std_init,
                  "activation": self.activation}
        return config

    def call(self, inputs, training=False, mask=None):
        # v_stream, a_stream = tf.split(inputs, 2, axis=1)  # TODO: only for graetz
        # v = self.v(v_stream)
        # a = self.a(a_stream)
        v = self.v(self.v_h(inputs))
        a = self.a(self.a_h(inputs))
        a_avg = tf.reduce_mean(a, axis=self.q_value_axis, keepdims=True, name='A_mean')
        # State-action values: Q(s, a) = V(s) + A(s, a) - A_mean(s, a)
        q = v + a - a_avg
        return q

    def reset_noise(self):
        if self.noise_std_init > 0:
            self.v_h.reset_noise()
            self.v.reset_noise()
            self.a_h.reset_noise()
            self.a.reset_noise()

    def set_noisy(self, active):
        if self.noise_std_init > 0:
            self.v_h.set_noisy(active)
            self.v.set_noisy(active)
            self.a_h.set_noisy(active)
            self.a.set_noisy(active)


class DistributionalDuelingQNetwork(QNetwork):
    """Dueling C51 Q head for Rainbow-style DQN.

    This is kept as a separate head so the original scalar Q heads above remain
    available. The layer returns action distributions with shape
    (batch, num_actions, num_atoms), or (batch, time, num_actions, num_atoms)
    for sequential stems.
    """

    def __init__(self, latent_v_dim=256, latent_a_dim=256, noise_std_init=0.5,
                 activation="relu", num_atoms=51, v_min=-10.0, v_max=30.0):
        super().__init__(name="distributional_dueling_Q_network")
        self.v_h_size = latent_v_dim
        self.a_h_size = latent_a_dim
        self.noise_std_init = noise_std_init
        self.activation = activation
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.v_h = None
        self.v = None
        self.a_h = None
        self.a = None

    def build(self, input_shape=None):
        self.check_initialization()

        if self.v_h_size is not None:
            if self.noise_std_init == 0:
                self.v_h = Dense(self.v_h_size, name='latent_V', activation=self.activation)
            else:
                self.v_h = NoisyDense(self.v_h_size, std_init=self.noise_std_init,
                                      name='latent_V', activation=self.activation)
        else:
            self.v_h = Layer()

        if self.noise_std_init == 0:
            self.v = Dense(self.num_atoms, name='V_atoms')
        else:
            self.v = NoisyDense(self.num_atoms, self.noise_std_init, name='V_atoms')

        if self.a_h_size is not None:
            if self.noise_std_init == 0:
                self.a_h = Dense(self.a_h_size, name='latent_A', activation=self.activation)
            else:
                self.a_h = NoisyDense(self.a_h_size, std_init=self.noise_std_init,
                                      name='latent_A', activation=self.activation)
        else:
            self.a_h = Layer()

        action_atom_count = self.num_actions * self.num_atoms
        if self.noise_std_init == 0:
            self.a = Dense(action_atom_count, name='A_atoms')
        else:
            self.a = NoisyDense(action_atom_count, self.noise_std_init, name='A_atoms')

        super(DistributionalDuelingQNetwork, self).build(input_shape)

    def get_config(self):
        config = {"latent_v_dim": self.v_h_size,
                  "latent_a_dim": self.a_h_size,
                  "noise_std_init": self.noise_std_init,
                  "activation": self.activation,
                  "num_atoms": self.num_atoms,
                  "v_min": self.v_min,
                  "v_max": self.v_max}
        return config

    def call(self, inputs, training=False, mask=None):
        v_logits_flat = self.v(self.v_h(inputs))
        a_logits_flat = self.a(self.a_h(inputs))

        batch_shape = tf.shape(a_logits_flat)[:-1]
        v_shape = tf.concat([batch_shape, [1, self.num_atoms]], axis=0)
        a_shape = tf.concat([batch_shape, [self.num_actions, self.num_atoms]], axis=0)

        v_logits = tf.reshape(v_logits_flat, v_shape)
        a_logits = tf.reshape(a_logits_flat, a_shape)
        a_avg = tf.reduce_mean(a_logits, axis=-2, keepdims=True, name='A_atom_mean')

        logits = v_logits + a_logits - a_avg
        return tf.nn.softmax(logits, axis=-1, name="c51_action_distributions")

    def get_support(self):
        return tf.linspace(float(self.v_min), float(self.v_max), int(self.num_atoms))

    def get_support_np(self):
        return np.linspace(float(self.v_min), float(self.v_max), int(self.num_atoms)).astype("float32")

    def reset_noise(self):
        if self.noise_std_init > 0:
            self.v_h.reset_noise()
            self.v.reset_noise()
            self.a_h.reset_noise()
            self.a.reset_noise()

    def set_noisy(self, active):
        if self.noise_std_init > 0:
            self.v_h.set_noisy(active)
            self.v.set_noisy(active)
            self.a_h.set_noisy(active)
            self.a.set_noisy(active)


class VanillaQNetwork(QNetwork):
    def __init__(self):
        super().__init__(name="default_Q_network")

    def build(self, input_shape=None):
        self.check_initialization()

        self.dense1 = Dense(128, name='latent_2')
        self.lrelu = LeakyReLU()
        self.dense2 = Dense(self.num_actions, name='Q')

    def get_config(self):
        return {}

    def call(self, inputs, training=None, mask=None):
        latent_q = self.lrelu(self.dense1(inputs))
        return self.dense2(latent_q)
