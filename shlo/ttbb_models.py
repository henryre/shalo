import numpy as np
import tensorflow as tf

from collections import defaultdict
from sklearn.decomposition import PCA
from shlo_base import SHLOModel
from utils import (
    map_words_to_symbols, symbol_embedding, SymbolTable, top_similarity
)


class TTBB(SHLOModel):
    """Implementation of A Simple but Tough-to-Beat-Baseline for Sent. Embedding
    In the basic model, the common component vector is computed before all
    computations. The embeddings are static, so no updates are made.
    """
    def __init__(self, embedding_file, marginals_file=None, name='TTBB',
                 save_file=None, n_threads=None):
        assert(embedding_file is not None)
        super(TTBB, self).__init__(
            name=name, embedding_file=embedding_file, save_file=save_file,
            n_threads=n_threads
        )
        # Get marginals file
        if marginals_file is not None:
            with open(marginals_file, 'rb') as f:
                self.word_freq = cPickle.load(f)
        else:
            self.word_freq = None

    def _static_common_component(self, tokens, U, p):
        """Compute the common component vector
            @tokens: list of lists of token ids representing sentences
            @U: matrix of word embeddings
            @p: marginal probability estimates for each word
        """
        self.a = self.train_kwargs.get('a', 0.01)
        X = []
        for t in tokens:
            if len(t) == 0:
                X.append(np.zeros(U.shape[1]))
                continue
            # Normalizer
            z = 1.0 / len(t)
            # Embed sentence
            q = (self.a / (self.a + p[t])).reshape((len(t), 1))
            X.append(z * np.sum(q * U[t, :], axis=0))
        # Compute first principal component
        X = np.array(X)
        pca = PCA(n_components=1, whiten=False, svd_solver='randomized')
        pca.fit(X)
        return np.ravel(pca.components_)

    def _preprocess_data(self, sentence_data, init=True, debug=False):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        # Just map tokens if not initializing
        if not init:
            return [
                np.ravel(
                    map_words_to_symbols(s, self.word_dict.lookup, self.ngrams)
                ) for s in sentence_data
            ]
        # If initializing, get marginal estimates and common component
        if self.word_freq is None:
            marginal_counts = defaultdict(int)
            tokens = []
            for s in sentence_data:
                t = np.ravel(
                    map_words_to_symbols(s, self.word_dict.lookup, self.ngrams)
                )
                tokens.append(t)
                for x in t:
                    marginal_counts[x] += 1
            # Estimate marginals
            self.marginals = np.zeros(self.word_dict.num_symbols())
            for k, v in marginal_counts.iteritems():
                self.marginals[k] = float(v)
            self.marginals /= sum(marginal_counts.values())
        else:

        # Compute sentence embeddings
        if debug:
            orig_a = self.train_kwargs.get('a')
            for a in np.logspace(-8, 2, num=21):
                self.train_kwargs['a'] = a
                msg = '== a={0} =='.format(a)
                print '\n{0}\n{1}\n{0}'.format('='*len(msg), msg)
                ccx = self._static_common_component(
                    tokens, symbol_embedding(self.embeddings), self.marginals
                )
                top_similarity(
                    symbol_embedding(self.embeddings),
                    self.word_dict.reverse(), 15, ccx
                )
            if orig_a is not None: self.train_kwargs['a'] = orig_a
        self.ccx = self._static_common_component(
            tokens, symbol_embedding(self.embeddings), self.marginals
        )
        self.train_tokens = tokens
        return tokens

    def _get_embedding(self):
        """
        Row 0 is 0 vector for no token
        Row 1 is 0 vector for unknown token
        Remaining rows are constant at pretrained emebdding
        """
        return tf.constant(
            symbol_embedding(self.embeddings),
            dtype=tf.float32, name='embedding_matrix'
        )

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


class TTBBTune(TTBB):
    """TTBB model with common component updated via gradient descent"""
    def __init__(self, embedding_file, marginals_file=None, name='TTBBTune',
                 save_file=None, n_threads=None):
        super(TTBBTune, self).__init__(
            embedding_file=embedding_file, marginals_file=marginals_file,
            name=name, save_file=save_file, n_threads=n_threads
        )

    def _get_embedding(self):
        """
        Row 0 is 0 vector for no token
        Row 1 is 0 vector for unknown token
        Remaining rows are constant at pretrained emebdding
        """
        return tf.Variable(
            symbol_embedding(self.embeddings),
            dtype=tf.float32, name='embedding_matrix'
        )

    def _get_a(self):
        return tf.Variable(self.a, dtype=tf.float32)

    def _get_common_component(self):
        return tf.Variable(self.ccx, dtype=tf.float32)


class TTBBTuneLazy(TTBB):
    """TTBB model with exact common component updates
    Common component vector updated after every epoch
    """
    def __init__(self, embedding_file, marginals_file=None, name='TTBBTuneLazy',
                 save_file=None, n_threads=None):
        super(TTBBTuneLazy, self).__init__(
            embedding_file=embedding_file, marginals_file=marginals_file,
            name=name, save_file=save_file, n_threads=n_threads
        )

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
        self.ccx = self._static_common_component(
            self.train_tokens, U, self.marginals
        )

    def _get_embedding(self):
        """
        Row 0 is 0 vector for no token
        Row 1 is 0 vector for unknown token
        Remaining rows are constant at pretrained emebdding
        """
        self.U = tf.Variable(
            symbol_embedding(self.embeddings),
            dtype=tf.float32, name='embedding_matrix'
        )
        return self.U

    def _get_a(self):
        return tf.Variable(self.a, dtype=tf.float32)

    def _get_common_component(self):
        self.ccx_placeholder = tf.placeholder(tf.float32, name='common_comp')
        return self.ccx_placeholder
