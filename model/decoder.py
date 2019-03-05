import tensorflow as tf


class Decoder():
    def __init__(self, name, hidden_size, embed_size, vocab_size, dropout_rate):
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
        self._dropout_rate = dropout_rate
        self._built = False
        if not self._built:
            self.__build_model()

    def __build_model(self):
        with tf.variable_scope(self._name, reuse=self._built):
            self._rnn_cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)
            if self._dropout_rate < 1.0:
                self._rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                    self._rnn_cell, output_keep_prob=self._dropout_rate)
            self._embeddings = tf.get_variable(
                "caption_embed", [self._vocab_size, self._embed_size], tf.float32)
        self._built = True
    
    def __call__(self, inputs, attrib_vectors, state, t):
        # Output for GRUCell = [batch_size, self.out_size],
        # [batch_size, hiddden_size]
        if not self._built:
            self.__build_model()
        if t == -1:
            vect_inputs = inputs
        else:
            vect_inputs = tf.nn.embedding_lookup(self._embeddings, inputs)
        if state is None:
            batch_size = tf.shape(inputs)[0]
            state = self._rnn_cell.zero_state(
                batch_size=batch_size, dtype=tf.float32)
        # Concat [batch_size, seq_len(one for static), embed_dim], [~, attr_emb_dim]
        if attrib_vectors is not None:
            vect_inputs = tf.concat([vect_inputs, attrib_vectors], axis=1)
        output, state = self._rnn_cell(vect_inputs, state)
        with tf.variable_scope("logits_dense", reuse=tf.AUTO_REUSE):
            output_l = tf.layers.dense(output, self._vocab_size)
        return output_l, state
    
    def dynamic_call(self, inputs, state, gen_mode=False):
        if not self._built:
            self.__build_model()
        vect_inputs = tf.nn.embedding_lookup(self._embeddings, inputs)
        outputs, final_state = tf.nn.dynamic_rnn(
            self._rnn_cell,
            inputs=vect_inputs,
            initial_state=state)
        if gen_mode:
            # only interested in the last output
            outputs = outputs[:, -1, :]
        outputs_r = tf.reshape(outputs, [-1, self._rnn_cell.output_size])
        with tf.variable_scope("logits_dense", reuse=tf.AUTO_REUSE):
            output_l = tf.layers.dense(outputs_r, units=self._vocab_size)
        output_l = tf.reshape(
            output_l, [tf.shape(outputs)[0], tf.shape(outputs)[1], -1])
        return output_l, final_state
    
    @property
    def rnn_cell(self):
        return self._rnn_cell