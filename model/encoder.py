import tensorflow as tf


# Encoder for sequence attributes
class Encoder():
    def __init__(self, name, hidden_size, embed_size, vocab_size):
        """
            Args:
                name: str, variable scope name
                hidden_size: int, hidden state dimensionality
                embed_size: int, embedding dimensionality
                vocab_size: int, size of vocabulary
        """
        self._name = name
        self._hidden_size = hidden_size
        self._embed_size = embed_size
        self._vocab_size = vocab_size
        self._built = False
        if not self._built:
            self.__build_model()
            self._built = True

    def __build_model(self):
        with tf.variable_scope(self._name, reuse=self._built):
            self._rnn_cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)
            # self._rnn_cell = tf.contrib.rnn.AttentionCellWrapper(
            #     _rnn_cell, self._inpu)
            self._embeddings = tf.get_variable(
                "seq_embed", [self._vocab_size, self._embed_size], tf.float32)
    
    def __call__(self, inputs):
        # Output for GRUCell = [batch_size, self.out_size],
        # [batch_size, hiddden_size]
        if not self._built:
            self.__build_model()
            self._built = True
        initial_state = self._rnn_cell.zero_state(
            batch_size=tf.shape(inputs)[0], dtype=tf.float32)
        vect_inputs = tf.nn.embedding_lookup(self._embeddings, inputs)
        # [batch_size, seq_len, hidden_dim]
        outputs, _ = tf.nn.dynamic_rnn(
            self.rnn_cell, vect_inputs, initial_state=initial_state)
        return outputs
    
    @property
    def rnn_cell(self):
        return self._rnn_cell