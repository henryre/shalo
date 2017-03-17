import numpy as np
import tensorflow as tf

from shalo_base import (
    SD, SHALOModel, SHALOModelFixed, SHALOModelPreTrain, SHALOModelRandInit
)
from utils import map_words_to_symbols, symbol_embedding, SymbolTable


class SHALOModelVectorMean(SHALOModel):
    """Base class for models with sentence embedding as mean of word vectors"""

    name = 'SHALOModelVectorMean'

    def _embed_sentences(self):
        """Mean of word vectors"""
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        s               = tf.reduce_sum(word_feats, axis=1)
        sv              = {'embeddings': word_embeddings}
        return s / tf.to_float(tf.reshape(self.input_lengths, (-1, 1))), sv


class LinearModel(SHALOModelVectorMean, SHALOModelFixed):
    """Linear model over pretrained embeddings"""

    name = 'LinearModel'

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


class fastText(SHALOModelVectorMean, SHALOModelRandInit):
    """Linear model over backprop-trained embeddings"""

    name = 'fastText'

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table
        if init:
            self.word_dict = SymbolTable()
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        return [
            map_words_to_symbols(s, mapper, self.ngrams) for s in sentence_data
        ]


class fastTextPreTrain(SHALOModelVectorMean, SHALOModelPreTrain):
    """Linear model over backprop-trained embeddings init. from pretrained"""

    name = 'fastTextPreTrain'

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with pre-embedded training words
        if init:
            self._word_table_init(sentence_data)
        # Process data
        mapper = self.word_dict.get if init else self.word_dict.lookup
        return [
            map_words_to_symbols(s, mapper, self.ngrams) for s in sentence_data
        ]
