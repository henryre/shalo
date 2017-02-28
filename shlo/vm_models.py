import numpy as np
import tensorflow as tf

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
        s = self.seed - 1
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
        s = self.seed - 1
        unk = tf.Variable(tf.random_normal((1, self.d), stddev=0.1, seed=s))
        pretrain = tf.Variable(self.embeddings, dtype=tf.float32)
        rem = tf.Variable(tf.random_normal(
            (self.word_dict.num_words()-len(self.embedding_words), self.d),
            stddev=0.1, seed=s
        ))
        return tf.concat(
            [zero, unk, pretrain, rem], axis=0, name='embedding_matrix'
        )
