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


class TFModel(object):

    def __init__(self, save_file=None, name='TFModel', n_threads=None):
        self.name       = name
        self.train_fn   = None
        self.loss       = None
        self.prediction = None
        self.save_dict  = None
        self.session    = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=n_threads,
                inter_op_parallelism_threads=n_threads
            )
        ) if n_threads is not None else tf.Session()
        if save_file is not None:
            self.load(save_file)

    def _build(self, **kwargs):
        raise NotImplementedError()

    def train(self, X, training_labels, **hyperparams):
        raise NotImplementedError()

    def predict(self, X, **kwargs):
        raise NotImplementedError()

    def score(self, X, **kwargs):
    	raise NotImplementedError()

    def save_info(self, model_name, **kwargs):
        pass

    def load_info(self, model_name, **kwargs):
        pass

    def save(self, model_name=None, verbose=True):
        """Save current TensorFlow model
            @model_name: save file names
            @verbose: be talkative?
        """
        model_name = model_name or self.name
        self.save_info(model_name)
        save_dict = self.save_dict or tf.global_variables()
        saver = tf.train.Saver(save_dict)
        saver.save(self.session, './' + model_name, global_step=0)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, verbose=True):
        """Load TensorFlow model from file
            @model_name: save file names
            @verbose: be talkative?
        """
        self.load_info(model_name)
        self._build()
        load_dict = self.save_dict or tf.global_variables()
        saver = tf.train.Saver(load_dict)
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name
            ))
