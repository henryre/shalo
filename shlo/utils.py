import numpy as np
import tensorflow as tf
import zlib


def scrub(text):
    """Clean text and force utf-8 encoding"""
    text = ''.join(c for c in text.lower() if ord(c) < 128)
    if isinstance(text, unicode):
        return text.encode('utf8')
    return unicode(text, 'utf8', errors='strict').encode('utf8')


def symbol_embedding(U):
    return np.vstack([np.zeros((2, U.shape[1])), U])


def map_words_to_symbols(words, mapper, ngrams=1, min_len=2):
    scrubbed_words = [scrub(word.lower()) for word in words]
    scrubbed_words = filter(lambda w: len(w) >= min_len, scrubbed_words)
    tokens, n = [], len(scrubbed_words)
    for i in xrange(n):
        for k in xrange(ngrams):
            if i + k + 1 > n:
                break
            tokens.append(mapper('~~'.join(scrubbed_words[i : i+k+1])))
    return tokens


def cos_sim(x, y, epsilon=1e-32):
    return np.dot(x, y) / (epsilon + np.linalg.norm(x) * np.linalg.norm(y))


def top_similarity(U, id2token, k, query_vector):
    sims = {}
    for i in xrange(U.shape[0]):
        sims[id2token[i]] = cos_sim(U[i, :], query_vector)
    top = sorted(((v, k) for k, v in sims.iteritems()), reverse=True)[:k]
    cut, title = 100, 'Most similar terms'
    top = [(x, y[:cut] + (len(y) > cut) * '...') for x, y in top]
    t = max(len(y) for x, y in top + [(0, title)]) + 5
    print "=" * 100
    print "{0}cos. sim.".format(title.ljust(t))
    print (len(title) * '-').ljust(t-1), '---------'
    print "\n".join("{0}{1:.4f}".format(y.ljust(t), x) for x, y in top)
    print "=" * 100


class FeatureHasher(object):
    """Simple implementation of the hashing trick"""

    def __init__(self, id_range=32000):
        self.id_range = id_range

    def hash(self, token):
        return zlib.adler32(scrub(token)) % self.id_range


class LabelBalancer(object):
    """Utility class to rebalance training labels
    For example, to get the indices of a training set
    with labels y and around 90 percent negative examples,
        LabelBalancer(y).get_train_idxs(rebalance=0.1)
    """

    def __init__(self, y):
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
        r = {v: k for k, v in self.d.iteritems()}
        r[0], r[1] = '~~NONE~~', '~~UNKNOWN~~'
        return r

    def num_words(self):
        return len(self.d)

    def num_symbols(self):
        return self.s
