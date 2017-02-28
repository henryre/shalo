import numpy as np
import tensorflow as tf

from collections import defaultdict
from sklearn.decomposition import PCA
from shlo_base import SHLOModel
from utils import scrub, symbol_embedding, SymbolTable


class VectorMeanSHLOModel(SHLOModel):

    def _get_embedding(self):
        raise NotImplementedError()

    def _embed_sentences(self):
        """Mean of word vectors"""
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        s               = tf.reduce_sum(word_feats, axis=1)
        return s / tf.to_float(tf.reshape(self.input_lengths, (-1, 1)))


class LinearModel(VectorMeanSHLOModel):

    def __init__(self, embedding_file, save_file=None, name='LinearModel',
                 n_threads=None):
        assert(embedding_file is not None)
        super(LinearModel, self).__init__(
            embedding_file, save_file, name, n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        return [
            [self.word_dict.lookup(scrub(word.lower())) for word in sentence]
            for sentence in sentence_data
        ]

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


class fastText(VectorMeanSHLOModel):

    def __init__(self, save_file=None, name='fastText', n_threads=None):
        super(fastText, self).__init__(None, save_file, name, n_threads)

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
        # Process data
        retrieve_fn = self.word_dict.get if init else self.word_dict.lookup
        return [
            [retrieve_fn(scrub(word.lower())) for word in sentence]
            for sentence in sentence_data
        ]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Initialize random embedding for UNKNOWN and all words
        """
        zero  = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed-1
        embed = tf.Variable(tf.random_normal(
            (self.word_dict.num_words() + 1, self.d), stddev=0.1, seed=s
        ))
        return tf.concat([zero, embed], axis=0, name='embedding_matrix')


class fastTextPreTrain(VectorMeanSHLOModel):

    def __init__(self, embedding_file, save_file=None, name='fastTextPreTrain',
                 n_threads=None):
        assert(embedding_file is not None)
        super(fastTextPreTrain, self).__init__(
            embedding_file, save_file, name, n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        retrieve_fn = self.word_dict.get if init else self.word_dict.lookup
        return [
            [retrieve_fn(scrub(word.lower())) for word in sentence]
            for sentence in sentence_data
        ]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Row 1 is random initialization for UNKNOWN
        Rows 2 : 2 + len(self.embedding_words) are pretrained initialization
        Remaining rows are random initialization
        """
        zero = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed-1
        unk = tf.Variable(tf.random_normal((1, self.d), stddev=0.1, seed=s))
        pretrain = tf.Variable(self.embeddings, dtype=tf.float32)
        rem = tf.Variable(tf.random_normal(
            (self.word_dict.num_words()-len(self.embedding_words), self.d),
            stddev=0.1, seed=s
        ))
        return tf.concat(
            [zero, unk, pretrain, rem], axis=0, name='embedding_matrix'
        )


class TTBB(SHLOModel):

    def __init__(self, embedding_file, a=0.01, save_file=None, name='TTBB',
                 n_threads=None):
        assert(embedding_file is not None)
        super(TTBB, self).__init__(
            embedding_file, save_file, name, n_threads
        )
        self.a = a

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        if not init:
            return [
                np.ravel([
                    self.word_dict.lookup(scrub(w.lower())) for w in s
                ]) for s in sentence_data
            ]
        marginal_counts = defaultdict(int)
        tokens = []
        for s in sentence_data:
            t = np.ravel([self.word_dict.lookup(scrub(w.lower())) for w in s])
            tokens.append(t)
            for x in t:
                marginal_counts[x] += 1
        # Estimate marginals
        self.marginals = np.zeros(self.word_dict.num_symbols())
        for k, v in marginal_counts.iteritems():
            self.marginals[k] = float(v)
        self.marginals /= sum(marginal_counts.values())
        # Compute sentence embeddings
        X = []
        U = symbol_embedding(self.embeddings)
        for t in tokens:
            if len(t) == 0:
                X.append(np.zeros(U.shape[1]))
                continue
            # Normalizer
            z = 1.0 / len(t)
            # Embed sentence
            q = (self.a / (self.a + self.marginals[t])).reshape((len(t), 1))
            X.append(z * np.sum(q * U[t, :], axis=0))
        # Compute first principal component
        X = np.array(X)
        pca = PCA(n_components=1, whiten=False, svd_solver='randomized')
        pca.fit(X)
        self.nonsense_projector = np.ravel(pca.components_)
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

    def _get_common_component(self):
        return tf.constant(self.nonsense_projector, dtype=tf.float32)

    def _embed_sentences(self):
        """Based on Arora et. al '17"""
        # Get word features
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        # Get marginal estimates and scaling term
        batch_size = tf.shape(word_feats)[0]
        p = tf.constant(self.marginals, dtype=tf.float32, name='marginals')
        q = tf.reshape(
            self.a / (self.a + tf.nn.embedding_lookup(p, self.input)),
            (batch_size, self.mx_len, 1)
        )
        # Compute initial sentence embedding
        z = tf.reshape(1.0 / tf.to_float(self.input_lengths), (batch_size, 1))
        S = z * tf.reduce_sum(q * word_feats, axis=1)
        # Common component removal
        ccx = tf.reshape(self._get_common_component(), (1, self.d))
        return S - tf.matmul(S, ccx * tf.transpose(ccx))


class TTBBTune(TTBB):

    def __init__(self, embedding_file, a=0.01, save_file=None, name='TTBBTune',
                 n_threads=None):
        super(TTBBTune, self).__init__(
            embedding_file, a, save_file, name, n_threads
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

    def _get_common_component(self):
        return tf.Variable(self.nonsense_projector, dtype=tf.float32)
