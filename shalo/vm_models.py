import numpy as np
import tensorflow as tf

from shalo_base import (
    SD, SHALOModel, SHALOModelFixed, SHALOModelPreTrain, SHALOModelRandInit
)
from utils import map_words_to_symbols, symbol_embedding, SymbolTable


class VectorMeanSHALOModel(SHALOModel):
    """Base class for models with sentence embedding as mean of word vectors"""

    def _embed_sentences(self):
        """Mean of word vectors"""
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        s               = tf.reduce_sum(word_feats, axis=1)
        sv              = {'embeddings': word_embeddings}
        return s / tf.to_float(tf.reshape(self.input_lengths, (-1, 1))), sv


class LinearModel(VectorMeanSHALOModel, SHALOModelFixed):
    """Linear model over pretrained embeddings"""
    def __init__(self, embedding_file, name='LinearModel', save_file=None,
                 n_threads=None):
        assert(embedding_file is not None)
        super(LinearModel, self).__init__(
            name=name, embedding_file=embedding_file, save_file=save_file,
            n_threads=n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        return [
            map_words_to_symbols(s, self.word_dict.lookup, self.ngrams)
            for s in sentence_data
        ]


class fastText(VectorMeanSHALOModel, SHALOModelRandInit):
    """Linear model over backprop-trained embeddings"""
    def __init__(self, name='fastText', save_file=None, n_threads=None):
        super(fastText, self).__init__(
            name=name, save_file=save_file, n_threads=n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table
        if init:
            self.word_dict = SymbolTable()
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        return [
            map_words_to_symbols(s, mapper, self.ngrams) for s in sentence_data
        ]


class fastTextPreTrain(VectorMeanSHALOModel, SHALOModelPreTrain):
    """Linear model over backprop-trained embeddings init. from pretrained"""
    def __init__(self, embedding_file, name='fastTextPreTrain', save_file=None,
                 n_threads=None):
        assert(embedding_file is not None)
        super(fastTextPreTrain, self).__init__(
            name=name, embedding_file=embedding_file, save_file=save_file,
            n_threads=n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        return [
            map_words_to_symbols(s, mapper, self.ngrams) for s in sentence_data
        ]
