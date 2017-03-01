import numpy as np

from shlo import (
    fastText, fastTextPreTrain, LinearModel, # Vector mean models
    TTBB, TTBBTune, TTBBTuneLazy,            # TTBB models
    LSTM, SparseLM,                 # Baseline models
)


SENNA  = 'data/senna.pkl'
DEPW2V = 'data/depw2v.pkl'
EMB    = SENNA
DIM    = 50 if EMB is SENNA else 300


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


def test_model(model, train, train_y, test, test_y, embedding=None):
    k = 1 if model in [LSTM, TTBB, TTBBTune, TTBBTuneLazy] else 2
    F = model(embedding) if embedding is not None else model()
    print "\n\nRunning test with [{0}]".format(F.name)
    F.train(
        train, train_y, 
        n_epochs=10, dim=DIM, ngrams=k,
        dev_sentence_data=test, dev_labels=test_y,
        seed=1701
    )
    F.score(test, test_y, verbose=True)


if __name__ == '__main__':
    # Get data
    train, train_y = get_data_from_file('data/train.txt')
    test, test_y   = get_data_from_file('data/test.txt')
    # Run test
    test_model(TTBBTuneLazy, train, train_y, test, test_y, EMB)
    test_model(TTBBTune, train, train_y, test, test_y, EMB)
    test_model(TTBB, train, train_y, test, test_y, EMB)
    test_model(fastTextPreTrain, train, train_y, test, test_y, EMB)
    test_model(fastText, train, train_y, test, test_y)
    test_model(LinearModel, train, train_y, test, test_y, EMB)
    test_model(LSTM, train, train_y, test, test_y)
    test_model(SparseLM, train, train_y, test, test_y)
