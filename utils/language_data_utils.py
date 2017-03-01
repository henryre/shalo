import cPickle
import numpy as np


DATA_DIR = 'data/'


def senna(vector_f='embeddings.txt', words_f ='words.lst', out_f='senna.pkl'):
    """Process raw Senna word vectors"""
    with open(DATA_DIR + words_f, 'rb') as f:
        words = [line.strip() for line in f]
    M = np.loadtxt(DATA_DIR + vector_f)
    print "Found {0} words".format(len(words))
    print "Found {0}x{1} embedding matrix".format(*M.shape)
    with open(DATA_DIR + out_f, 'wb') as f:
        cPickle.dump((words, M), f)


def dep_w2v(data_fname='deps.words', out_fname='depw2v.pkl'):
    """Process raw dependency word2vec data from Levy & Goldberg '14"""
    M = np.loadtxt(DATA_DIR + data_fname, converters={0: lambda x: 0})
    M = M[:, 1:]
    print "Loaded {0}x{1} word vector matrix".format(*M.shape)
    with open(DATA_DIR + data_fname, 'rb') as f:
        words = [line.split()[0] for line in f]
    with open(DATA_DIR + out_fname, 'wb') as f:
        cPickle.dump((words, M), f)


def glove(data_fname='glove.840B.300d.txt', out_fname='glove.pkl'):
    """Process raw dependency GloVe data from Socher '13"""
    words, U, dim = [], [], None
    with open(DATA_DIR + data_fname, 'rb') as f:
        for j, line in enumerate(f):
            x = line.strip().split()
            word, vector, d = x[0], np.ravel(x[1:]), len(x) - 1
            if dim is None: dim = d
            elif d != dim:  raise Exception('{0}: {1}!={2}'.format(j, dim, d))
            U.append(vector)
            words.append(word)
    U = np.array(U)
    print "Found {0} words".format(len(words))
    print "Found {0}x{1} embedding matrix".format(*U.shape)
    with open(DATA_DIR + out_fname, 'wb') as f:
        cPickle.dump((words, U), f)


def word_freq(data_fname='count_1w.txt', out_fname='word_freq.pkl'):
    counts, s = {}, 0
    with open(DATA_DIR + data_fname, 'rb') as f:
        for line in f:
            word, ct = line.strip().split()
            ct = int(ct)
            counts[word] = ct
            s += ct
    for k, v in counts.iteritems():
        counts[k] = float(v) / s
    with open(DATA_DIR + out_fname, 'wb') as f:
        cPickle.dump(counts, f)
