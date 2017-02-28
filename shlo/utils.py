import cPickle
import numpy as np
import tensorflow as tf
import zlib


DATA_DIR = 'data/'


def senna(vector_f='embeddings.txt', words_f ='words.lst', out_f='senna.pkl'):
    with open(DATA_DIR + words_f, 'rb') as f:
        words = [line.strip() for line in f]
    M = np.loadtxt(DATA_DIR + vector_f)
    print "Found {0} words".format(len(words))
    print "Found {0}x{1} embedding matrix".format(*M.shape)
    with open(DATA_DIR + out_f, 'wb') as f:
        cPickle.dump((words, M), f)


def dep_w2v(data_fname='deps.words', out_fname='depw2v.pkl'):
    M = np.loadtxt(DATA_DIR + data_fname, converters={0: lambda x: 0})
    M = M[:, 1:]
    print "Loaded {0}x{1} word vector matrix".format(*M.shape)
    with open(DATA_DIR + data_fname, 'rb') as f:
        words = [line.split()[0] for line in f]
    with open(DATA_DIR + out_fname, 'wb') as f:
        cPickle.dump((words, M), f)

def scrub(text):
    text = ''.join(c for c in text.lower() if ord(c) < 128)
    if isinstance(text, unicode):
        return text.encode('utf8')
    return unicode(text, 'utf8', errors='strict').encode('utf8')


def symbol_embedding(U):
    return np.vstack([np.zeros((2, U.shape[1])), U])


class FeatureHasher(object):

    def __init__(self, id_range=32000):
        self.id_range = id_range

    def hash(self, token):
        return zlib.adler32(scrub(token)) % self.id_range



class LabelBalancer(object):
    def __init__(self, y):
        """Utility class to rebalance training labels
        For example, to get the indices of a training set
        with labels y and around 90 percent negative examples,
            LabelBalancer(y).get_train_idxs(rebalance=0.1)
        """
        self.y = np.ravel(y)
    
    def _get_pos(self, split):
        return np.where(self.y > (split + 1e-6))[0]

    def _get_neg(self, split):
        return np.where(self.y < (split - 1e-6))[0]
    
    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0-pn)))
        s = int(round(float((1.0-pn) * n) / pn))
        return (m,r) if r <= n else ((s,n) if s <= m else (m,n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0-frac_pos)[::-1]

    def get_train_idxs(self, rebalance=False, split=0.5):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, fraction is 0.5. If False, no balancing.
            @split: Split point for positive and negative classes
        """
        pos, neg = self._get_pos(split), self._get_neg(split)
        if rebalance:
            p = 0.5 if rebalance == True else rebalance
            n_neg, n_pos = self._get_counts(len(neg), len(pos), p)
            pos = np.random.choice(pos, size=n_pos, replace=False)
            neg = np.random.choice(neg, size=n_neg, replace=False)
        idxs = np.concatenate([pos, neg])
        np.random.shuffle(idxs)
        return idxs


class SymbolTable:
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1): 
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w, strict=False):
        return self.d[w] if strict else self.d.get(w, self.unknown)

    def reverse(self):
        return {v: k for k, v in self.d.iteritems()}

    def num_words(self):
        return len(self.d)

    def num_symbols(self):
        return self.s
