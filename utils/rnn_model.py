from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    # if using GRU Cells
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)



def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
      """Makes a RNN cell from the given hyperparameters.
      Args:
        rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
            RNN.
        dropout_keep_prob: The float probability to keep the output of any given
            sub-cell.
        attn_length: The size of the attention vector.
        base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
      Returns:
          A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
      """
      cells = []
      for num_units in rnn_layer_sizes:
        cell = base_cell(num_units)
        if attn_length and not cells:
          # Add attention wrapper to first layer.
          cell = tf.contrib.rnn.AttentionCellWrapper(
              cell, attn_length, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)

      cell = tf.contrib.rnn.MultiRNNCell(cells)

      return cell
