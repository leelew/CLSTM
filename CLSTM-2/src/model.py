from datetime import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers


class CausalLSTMNodeCell(layers.Layer):
    """Causal LSTM for each node on the causality-structure.
    
    The calculation process is shown as following:
    1. horizontal forward process, used hidden state and cell state from 
       last timestep to generate hidden and cell state now timstep.
    2. vertical forward process, used inputs, hidden state 
       from last timestep and now timestep.
    """

    def __init__(self,
                 num_hiddens=16,
                 num_children=3):
        super(CausalLSTMNodeCell, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_children = num_children

        # horiziontal forward, i.e., standard LSTM
        self._ifo_x = layers.Dense(3 * num_hiddens, activation=None)
        self._ifo_h = layers.Dense(3 * num_hiddens, activation=None)
        self._a_x = layers.Dense(num_hiddens, activation=None)
        self._a_h = layers.Dense(num_hiddens, activation=None)

        # vertical forward
        if num_children == 0:
            pass  # print()#'This is a leaf node')
        else:
            self._r_child_x = layers.Dense(
                num_children * num_hiddens, activation=None)
            self._r_child_h = layers.Dense(
                num_children * num_hiddens, activation=None)
        self._n_1_x = layers.Dense(num_hiddens, activation=None)
        self._n_1_h = layers.Dense(num_hiddens, activation=None)
        self._n_2_x = layers.Dense(num_hiddens, activation=None)
        self._n_2_h = layers.Dense(num_hiddens, activation=None)

    def _horizontal_forward(self, inputs, h_prev, c_prev):
        """forward pass of horizontal pass."""
        # generate input, forget, output gates
        ifo = tf.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = tf.split(ifo, 3, axis=-1)

        # generate current information state
        a = tf.math.tanh(self._a_x(inputs)+self._a_h(h_prev))

        # generate current cell state
        c = tf.math.multiply(i, a) + tf.math.multiply(f, c_prev)

        # generate current hidden state
        h = tf.math.multiply(o, tf.math.tanh(c))

        return h, c

    def _vertical_forward(self, inputs, h_prev, h, child_n=None):
        """forward pass of vertical pass"""
        if self.num_children == 0:
            r = 0
        else:
            # generate intermediate variable for neighborhood
            # (None, num_hiddens * num_child)
            child_r = tf.sigmoid(self._r_child_x(
                inputs) + self._r_child_h(h_prev))
            # (num_child, None, num_hiddens)
            child_r = tf.reshape(
                child_r, [self.num_children, -1, self.num_hiddens])

            # (num_child, None, num_hiddens)
            child_r_n = tf.math.multiply(child_r, child_n)
            r = tf.reduce_sum(child_r_n, axis=0)  # (None, num_hiddens)

        # generate weight for neighborhood and hidden state
        n_1 = tf.sigmoid(self._n_1_x(inputs) + self._n_1_h(h_prev))
        n_2 = tf.sigmoid(self._n_2_x(inputs) + self._n_2_h(h_prev))

        # generate current neighborhood state
        n = tf.math.multiply(n_1, r) + tf.math.multiply(n_2, h)

        return n

    def call(self, inputs, h_prev, c_prev, child_n=None):
        h, c = self._horizontal_forward(inputs, h_prev, c_prev)
        n = self._vertical_forward(inputs, h_prev, h, child_n)

        return n, h, c


class CausalLSTMCell(tf.keras.layers.Layer):
    def __init__(self,
                 num_hiddens=16,
                 num_nodes=6,
                 num_child_nodes=None,
                 input_idx=None,
                 child_state_idx=None):
        super(CausalLSTMCell, self).__init__()

        self.num_hiddens = num_hiddens
        self.num_nodes = num_nodes
        self.input_idx = input_idx 
        self.child_state_idx = child_state_idx
        self.children = num_child_nodes 

        self.lstm = []
        for i in range(num_nodes):
            self.lstm.append(CausalLSTMNodeCell(num_hiddens, num_child_nodes[i]))

    def update_state(self, state, state_new, state_idx, N):

        if state_idx == 0:
            state = tf.concat([state_new, state[state_idx+1:, :, :]], axis=0)
        elif state_idx == N:
            state = tf.concat([state[:state_idx, :, :], state_new], axis=0)
        else:
            state = tf.concat(
                [state[:state_idx, :, :], state_new, state[state_idx+1:, :, :]], axis=0)

        return state

    def call(self, inputs, h, c, n):

        # inputs: [None, features]
        # h, c: [num_nodes, None, num_hiddens]
        # n: [num_nodes, None, num_hiddens]

        for i in range(self.num_nodes):
            _in_x = tf.stack([inputs[:, k] for k in self.input_idx[i]], axis=-1)
            _h, _c = h[i, :, :], c[i, :, :]  # (None, num_hiddens)

            if self.children[i] == 0:
                n_new, h_new, c_new = self.lstm[i](_in_x, _h, _c, child_n=None)

            else:
                # (num_children, None, num_hiddens)
                child_n = tf.stack([n[j, :, :] for j in self.child_state_idx[i]], axis=0)
                n_new, h_new, c_new = self.lstm[i](_in_x, _h, _c, child_n)

            n = self.update_state(
                n, n_new[tf.newaxis, :, :], i, self.num_nodes)
            h = self.update_state(
                h, h_new[tf.newaxis, :, :], i, self.num_nodes)
            c = self.update_state(
                c, c_new[tf.newaxis, :, :], i, self.num_nodes)

        return n, h, c


class CausalLSTM(tf.keras.Model):
    def __init__(self,
                 num_child_nodes,
                 input_idx,
                 child_state_idx,
                 num_nodes,
                 num_hiddens=16,
                 input_len=10,
                 batch_size=50,
                 lag_time=0,
                 ):
        super(CausalLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.t_x = input_len
        self.batch_size = batch_size
        self.lag_time = lag_time

        self.clstm = []
        print(self.t_x)
        for i in range(self.t_x):
            self.clstm.append(CausalLSTMCell(
                num_hiddens, num_nodes, num_child_nodes, input_idx, child_state_idx))

        self.dense = layers.Dense(1)

    def initial_tree_state(self):
        """initial tree state using default LSTM."""
        initializer = tf.keras.initializers.Zeros()  # RandomUniform(minval=0., maxval=1.)
        h0 = initializer(
            shape=(self.num_nodes, self.batch_size, self.num_hiddens))
        c0 = initializer(
            shape=(self.num_nodes, self.batch_size, self.num_hiddens))
        n0 = initializer(
            shape=(self.num_nodes, self.batch_size, self.num_hiddens))

        return n0, h0, c0

    def call(self, inputs):
        """Causality LSTM"""
        n, h, c = self.initial_tree_state()

        for i in range(self.t_x):
            if i-self.lag_time < 0:
                k = i
            else:
                k = i-self.lag_time
            n, h, c = self.clstm[i](inputs[:, k, :], h, c, n)

        out = self.dense(n[-1, :, :])  # output = tf.stack(output, axis=1)

        return out


class ForestCausalLSTM(tf.keras.Model):

    def __init__(
        self, 
        group_num_child_nodes, 
        group_input_idx, 
        group_child_state_idx,
        num_hiddens=16, 
        batch_size=50,
        input_len=10,
        ):
        super().__init__()
        
        self.group_num_child_nodes = group_num_child_nodes
        self.group_input_idx = group_input_idx
        self.group_child_state_idx = group_child_state_idx
        
        self.batch_size = batch_size
        self.input_len = input_len
        self.num_hiddens = num_hiddens

    def __call__(self, inputs):
        for timestep in group_num_child_nodes.keys():
            num_child_nodes = self.group_num_child_nodes[timestep]
            input_idx = self.group_input_idx[timestep]
            child_state_idx = self.group_child_state_idx[timestep]

            num_nodes = len(num_child_nodes)
            
            out = CausalLSTM(num_nodes,
                            num_hiddens,
                            children,
                            child_input_idx,
                            child_state_idx,
                            input_len,
                            batch_size=32)
