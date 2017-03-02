import cPickle
import numpy as np
import tensorflow as tf

from collections import Counter
from shalo_base import SD, SHALOModel, SHALOModelFixed, SHALOModelPreTrain
from sklearn.decomposition import PCA
from utils import (
    map_words_to_symbols, symbol_embedding, SymbolTable, top_similarity
)


class TTBB(SHALOModelFixed):
    """Implementation of A Simple but Tough-to-Beat-Baseline for Sent. Embedding
    In the basic model, the common component vector is computed before all
    computations. The embeddings are static, so no updates are made.
    """

    name = 'TTBB'

    def __init__(self, embedding_file, word_freq_file=None, save_file=None,
                 n_threads=None):
        SHALOModelFixed.__init__(self, embedding_file, save_file, n_threads)
        # Get marginals file
        self._get_word_freq(word_freq_file)

    def _get_word_freq(self, fname):
        self.word_freq = None
        if fname is not None:
            with open(fname, 'rb') as f:
                self.word_freq = cPickle.load(f)

    def _static_common_component(self, tokens, U, p, a):
        """Compute the common component vector
            @tokens: list of lists of token ids representing sentences
            @U: matrix of word embeddings
            @p: marginal probability estimates for each word
        """
        X = []
        for t in tokens:
            if len(t) == 0:
                X.append(np.zeros(U.shape[1]))
                continue
            # Normalizer
            z = 1.0 / len(t)
            # Embed sentence
            q = (a / (a + p[t])).reshape((len(t), 1))
            X.append(z * np.sum(q * U[t, :], axis=0))
        # Compute first principal component
        X = np.array(X)
        pca = PCA(n_components=1, whiten=False, svd_solver='randomized')
        pca.fit(X)
        return np.ravel(pca.components_)

    def explore_common_component(self, tokens, a_rng=None):
        for a in (a_rng or np.logspace(-8, 2, num=21)):
            msg = '== a={0} =='.format(a)
            print '\n{0}\n{1}\n{0}'.format('='*len(msg), msg)
            ccx = self._static_common_component(
                tokens, symbol_embedding(self.embeddings), self.marginals, a
            )
            top_similarity(
                symbol_embedding(self.embeddings),
                self.word_dict.reverse(), 15, ccx
            )

    def _preprocess_data(self, sentence_data, init=True, debug=False):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        # Map tokens and return if not initializing
        tokens = [
            np.ravel(
                map_words_to_symbols(s, self.word_dict.lookup, self.ngrams)
            ) for s in sentence_data
        ]
        self.train_tokens = tokens
        if not init:
            return tokens
        # If initializing, get marginal estimates 
        if self.word_freq is None:
            marginal_counts = Counter([w for t in tokens for w in t])
            self.marginals = np.zeros(self.word_dict.num_symbols())
            for k, v in marginal_counts.iteritems():
                self.marginals[k] = float(v)
            self.marginals /= sum(marginal_counts.values())
        else:
            self.marginals = np.zeros(self.word_dict.num_symbols())
            for word, idx in self.word_dict.d.iteritems():
                self.marginals[idx] = self.word_freq.get(word, 0.0)
        # Compute common component
        if debug:
            self.explore_common_component(tokens)
        self.a = self.train_kwargs.get('a', 0.01)
        self.ccx = self._static_common_component(
            tokens, symbol_embedding(self.embeddings), self.marginals, self.a
        )
        return tokens

    def _get_a(self):
        return tf.constant(self.a, dtype=tf.float32)

    def _get_common_component(self):
        return tf.constant(self.ccx, dtype=tf.float32)

    def _embed_sentences(self):
        """Tensorflow version of @_static_common_component"""
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
        self._get_word_freq(word_freq_file)

    def _epoch_post_process(self, t, debug=False):
        # Update the common component
        if debug and ((t+1) % 5 == 0):
            a, ccx = self.session.run([self.a_var, self.ccx_var])
            print a
            top_similarity(
                symbol_embedding(self.embeddings),
                self.word_dict.reverse(), 15, ccx
            )

    def _get_a(self):
        self.a_var = tf.Variable(self.a, dtype=tf.float32)
        return self.a_var

    def _get_common_component(self):
        self.ccx_var = tf.Variable(self.ccx, dtype=tf.float32)
        return self.ccx_var


class TTBBTuneLazy(SHALOModelPreTrain, TTBB):
    """TTBB model with exact common component updates
    Common component vector updated after every epoch
    """

    name = 'TTBBTuneLazy'

    def __init__(self, embedding_file, word_freq_file=None, save_file=None,
                 n_threads=None):
        SHALOModelPreTrain.__init__(self, embedding_file, save_file, n_threads)
        # Get marginals file
        self._get_word_freq(word_freq_file)

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
        U = self.session.run(self.U)
        a = self.session.run(self.a_var)
        self.ccx = self._static_common_component(
            self.train_tokens, U, self.marginals, a
        )

    def _get_a(self):
        self.a_var = tf.Variable(self.a, dtype=tf.float32)
        return self.a_var

    def _get_common_component(self):
        self.ccx_placeholder = tf.placeholder(tf.float32, name='common_comp')
        return self.ccx_placeholder
