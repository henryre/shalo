import numpy as np
import os
import sys

from collections import OrderedDict
from shalo import *


SENNA  = 'data/senna.pkl'
DEPW2V = 'data/depw2v.pkl'
FREQ   = 'data/word_freq.pkl'
EMB    = SENNA if 'CI' in os.environ else DEPW2V
DIM    = 50 if EMB is SENNA else 300


MODELS = OrderedDict([
    ('sparselm', SparseLinearModel), ('lstmpretrain', LSTMPreTrain),
    ('linearmodel', LinearModel), 
    ('fasttext', fastText), ('fasttextpretrain', fastTextPreTrain),
    ('ttbb', TTBB), ('ttbbtune', TTBBTune),
    ('ttbbtuneexact', TTBBTuneExact), ('ttbbtunelazy', TTBBTuneLazy),
])

NEED_EMBED = [
    LinearModel, LSTMPreTrain, fastTextPreTrain,
    TTBB, TTBBTune, TTBBTuneExact, TTBBTuneLazy
]
NEED_FREQ  = [TTBB, TTBBTune, TTBBTuneExact, TTBBTuneLazy]


def get_data_from_file(fname):
    labels, sentences = [], []
    with open(fname, 'rb') as f:
        for line in f:
            label, text = line.strip().split(' ', 1)
            text = text.split(' ')
            labels.append((int(label) + 1) / 2)
            sentences.append(text)
    labels = np.ravel(labels)
    return sentences, labels


def test_model(model, train, train_y, test, test_y):
    kwargs = {}
    if model in NEED_EMBED:
        kwargs['embedding_file'] = EMB
    if model in NEED_FREQ:
        kwargs['word_freq_file'] = FREQ
    F = model(**kwargs)
    print "\n\nRunning test with [{0}]".format(F.name)
    ngrams   = 1 if model in [LSTM, LSTMPreTrain] else 2
    n_epochs = 3 if 'CI' in os.environ else 20
    F.train(
        train, train_y, 
        n_epochs=n_epochs, dim=DIM, ngrams=ngrams,
        dev_sentence_data=test, dev_labels=test_y,
        seed=1701
    )
    F.score(test, test_y, verbose=True)


if __name__ == '__main__':
    # Get data
    train, train_y = get_data_from_file('data/train.txt')
    test, test_y   = get_data_from_file('data/test.txt')
    # Run test
    if len(sys.argv) == 1:
        for model in MODELS.values():
            test_model(model, train, train_y, test, test_y)
    elif sys.argv[1].lower() in MODELS:
        test_model(MODELS[sys.argv[1].lower()], train, train_y, test, test_y)
