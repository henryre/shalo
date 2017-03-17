import cPickle
import numpy as np
import tensorflow as tf

from shalo_base import SD, SHALOModel, SHALOModelFixed, SHALOModelPreTrain
from sklearn.decomposition import PCA
from utils import (
    GRAMSEP, map_words_to_symbols, symbol_embedding, SymbolTable, top_similarity
)


class TTBB(SHALOModelFixed):
    """Implementation of A Simple but Tough-to-Beat Baseline for Sent. Embedding
    In the basic model, the common component vector is computed before all
    computations. The embeddings are static, so no updates are made.
    """

    name = 'TTBB'

    def __init__(self, embedding_file, word_freq_file, save_file=None,
                 n_threads=None):
        SHALOModelFixed.__init__(self, embedding_file, save_file, n_threads)
        # Get marginals file
        with open(word_freq_file, 'rb') as f:
            self.word_freq = cPickle.load(f)

    def _word_table_init(self, training_sentences):
        self.word_dict = SymbolTable()
        for word in self.embedding_words:
            self.word_dict.get(word)

    def _get_mapper(self, init):
        return self.word_dict.lookup

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self._word_table_init(sentence_data)
        # Process data
        # Map tokens and return if not initializing
        mapper = self._get_mapper(init)
        tokens = [
            np.ravel(map_words_to_symbols(s, mapper, self.ngrams))
            for s in sentence_data
        ]
        self.train_tokens = tokens
        if not init:
            return tokens
        # If initializing, get marginal estimates 
        self.marginals = np.zeros(self.word_dict.num_symbols())
        for word, idx in self.word_dict.d.iteritems():
            # Try getting word frequency directly
            if word in self.word_freq:
                self.marginals[idx] = self.word_freq[word]
            # Otherwise, try getting minimum frequency among sub-grams
            split_grams = word.split(GRAMSEP)
            if len(split_grams) > 1:
                min_freq = min(self.word_freq.get(w, 0.0) for w in split_grams)
                self.marginals[idx] = min_freq
        # Get initial smoother value
        self.a = self.train_kwargs.get('a', 0.01)
        return tokens

    def _compute_train_common_component(self, init=False):
        if init:
            self.session.run(tf.global_variables_initializer())
        x_array, x_len = self._get_data_batch(self.train_tokens)
        self.ccx = self.session.run(self.tf_ccx, {
            self.input:         x_array,
            self.input_lengths: x_len    
        })
        return self.ccx

    def _get_a(self):
        return tf.constant(self.a, dtype=tf.float32)

    def _get_common_component(self):
        self.ccx = self._compute_train_common_component(init=True)
        return tf.constant(self.ccx, dtype=tf.float32)

    def _embed_sentences(self):
        """Tensorflow implementation of Simple but Tough-to-Beat Baseline"""
        # Get word features
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        # Get marginal estimates and scaling term
        batch_size = tf.shape(word_feats)[0]
        a = self._get_a()
        p = tf.constant(self.marginals, dtype=tf.float32, name='marginals')
        q = tf.reshape(
            a / (a + tf.nn.embedding_lookup(p, self.input)),
            (batch_size, self.mx_len, 1)
        )
        # Compute initial sentence embedding
        z = tf.reshape(1.0 / tf.to_float(self.input_lengths), (batch_size, 1))
        S = z * tf.reduce_sum(q * word_feats, axis=1)
        # Compute common component
        S_centered = S - tf.reduce_mean(S, axis=0)
        _, _, V = tf.svd(S_centered, full_matrices=False, compute_uv=True)
        self.tf_ccx = tf.stop_gradient(tf.gather(tf.transpose(V), 0))
        # Common component removal
        ccx = tf.reshape(self._get_common_component(), (1, self.d))
        sv = {'embeddings': word_embeddings, 'a': a, 'p': p, 'ccx': ccx}
        return S - tf.matmul(S, ccx * tf.transpose(ccx)), sv


class TTBBTune(SHALOModelPreTrain, TTBB):
    """TTBB model with common component updated via gradient descent"""

    name = 'TTBBTune'

    def __init__(self, embedding_file, word_freq_file=None, save_file=None,
                 n_threads=None):
        SHALOModelPreTrain.__init__(self, embedding_file, save_file, n_threads)
        # Get marginals file
        with open(word_freq_file, 'rb') as f:
            self.word_freq = cPickle.load(f)

    def _get_mapper(self, init):
        return self.word_dict.get if init else self.word_dict.lookup

    def _epoch_post_process(self, t, debug=False):
        # Explore common component in debug mode
        if debug and ((t+1) % 5 == 0):
            U, a, ccx = self.session.run([self.U, self.a_var, self.ccx_var])
            print a
            top_similarity(U, self.word_dict.reverse(), 15, ccx)

    def _get_a(self):
        self.a_var = tf.Variable(self.a, dtype=tf.float32)
        return self.a_var

    def _get_common_component(self):
        self.ccx = self._compute_train_common_component(init=True)
        self.ccx_var = tf.Variable(self.ccx, dtype=tf.float32)
        return self.ccx_var


class TTBBTuneExact(TTBBTune):
    """TTBB model with common component updated via gradient descent"""

    name = 'TTBBTuneExact'

    def _get_common_component(self):
        return self.tf_ccx


class TTBBTuneLazy(TTBBTune):
    """TTBB model with exact common component updates
    Common component vector updated after every epoch
    """

    name = 'TTBBTuneLazy'

    def _get_feed(self, x_batch, len_batch, y_batch=None):
        feed = {
            self.input:           x_batch, 
            self.input_lengths:   len_batch,
            self.ccx_placeholder: self.ccx,
        }
        if y_batch is not None:
            feed[self.y] = y_batch
        return feed

    def _epoch_post_process(self, t):
        # Update the common component
        self.ccx = self._compute_train_common_component(init=False)

    def _get_common_component(self):
        self.ccx = self._compute_train_common_component(init=True)
        self.ccx_placeholder = tf.placeholder(tf.float32, name='common_comp')
        return self.ccx_placeholder
