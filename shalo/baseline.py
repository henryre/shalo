import numpy as np
import scipy.sparse as sparse
import tensorflow as tf

from shalo_base import SHALOModel
from utils import map_words_to_symbols, symbol_embedding, SymbolTable


class SparseLM(SHALOModel):
    """Sparse linear model over BOW indicator vector"""
    def __init__(self, name='SparseLM', save_file=None, n_threads=None):
        super(SparseLM, self).__init__(
            name=name, save_file=save_file, n_threads=n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        tokens = [
            map_words_to_symbols(s, mapper, self.ngrams) for s in sentence_data
        ]
        if init:
            self.td = self.word_dict.num_symbols()
        return tokens

    def _get_data_batch(self, x_batch):
        # Construct LIL matrix
        X_lil = sparse.lil_matrix((len(x_batch), self.td))
        for j, x in enumerate(x_batch):
            for t in x:
                X_lil[j, t] += 1
        # Get batch data
        indices, ids, weights = [], [], []
        max_len = 0
        for i, (row, data) in enumerate(zip(X_lil.rows, X_lil.data)):
            # Dummy weight for all-zero row
            if len(row) == 0:
                indices.append((i, 0))
                ids.append(0)
                weights.append(0.0)
                continue
            # Update indices by position
            max_len = max(max_len, len(row))
            indices.extend((i, t) for t in xrange(len(row)))
            ids.extend(row)
            weights.extend(data)
        shape = (len(X_lil.rows), max_len)
        return [indices, shape, ids, weights], None

    def _get_feed(self, x_batch, len_batch, y_batch=None):
        indices, shape, ids, weights = x_batch
        feed = {
            self.indices: indices,
            self.shape:   shape,
            self.ids:     ids,
            self.weights: weights,
        }
        if y_batch is not None:
            feed[self.y] = y_batch
        return feed

    def _build(self):
        assert(self.lr is not None)
        assert(self.l2_penalty is not None)
        assert(self.loss_function is not None)
        # Define input placeholders
        self.indices = tf.placeholder(tf.int64) 
        self.shape   = tf.placeholder(tf.int64, (2,))
        self.ids     = tf.placeholder(tf.int64)
        self.weights = tf.placeholder(tf.float32)
        self.y       = tf.placeholder(tf.float32, (None,))
        # Define training variables
        sparse_ids = tf.SparseTensor(self.indices, self.ids, self.shape)
        sparse_vals = tf.SparseTensor(self.indices, self.weights, self.shape)
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        w = tf.Variable(tf.random_normal((self.td, 1), stddev=0.01, seed=s1))
        b = tf.Variable(tf.random_normal((1, 1), stddev=0.01, seed=s2))
        z = tf.nn.embedding_lookup_sparse(
            params=w, sp_ids=sparse_ids, sp_weights=sparse_vals, combiner='sum'
        )
        h = tf.squeeze(tf.add(z, b))
        # Define training procedure
        self.loss       = self._get_loss(h, self.y)
        self.loss      += self.l2_penalty * tf.nn.l2_loss(w)
        self.prediction = tf.sigmoid(h)
        self.train_fn   = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.save_dict  = self._get_save_dict(w=w, b=b)


def get_rnn_output(output, dim, lengths):
    """Get last output of RNN"""
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    index = tf.range(0, batch_size) * max_length + (lengths - 1)
    flat = tf.reshape(output, [-1, dim])
    h = tf.gather(flat, index)
    return h


class LSTM(SHALOModel):
    """Simple LSTM for sequence classification"""
    def __init__(self, name='LSTM', save_file=None, n_threads=None):
        super(LSTM, self).__init__(
            name=name, save_file=save_file, n_threads=n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        return [
            map_words_to_symbols(s, mapper, 1, 0) for s in sentence_data
        ]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Initialize random embedding for UNKNOWN and all words
        """
        zero  = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed - 1
        embed = tf.Variable(tf.random_normal(
            (self.word_dict.num_words() + 1, self.d), stddev=0.1, seed=s
        ))
        return tf.concat([zero, embed], axis=0, name='embedding_matrix')

    def _get_save_dict(self, **kwargs):
        return None

    def _embed_sentences(self):
        """Embed sentences via the last output cell of an LSTM"""
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        batch_size      = tf.shape(self.input)[0]
        with tf.variable_scope("LSTM") as scope:
            tf.set_random_seed(self.seed - 1)
            # LSTM architecture
            cell = tf.contrib.rnn.BasicLSTMCell(self.d)
            # Set RNN
            initial_state = cell.zero_state(batch_size, tf.float32)
            rnn_out, _ = tf.nn.dynamic_rnn(
                cell, word_feats, sequence_length=self.input_lengths,
                initial_state=initial_state, time_major=False               
            )
        # Get potentials
        return get_rnn_output(rnn_out, self.d, self.input_lengths), {}
