import numpy as np

from shlo import (
    fastText, fastTextPreTrain, LinearModel, TTBB, TTBBTune, # SHLO models
    LSTM, SparseLinearModel,                                 # Baseline models
)


SENNA = 'data/senna.pkl'


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
    F = model(embedding) if embedding is not None else model()
    print "\n\nRunning test with [{0}]".format(F.name)
    F.train(
        train, train_y, 
        n_epochs=10,
        dev_sentence_data=test, dev_labels=test_y,
        seed=1701
    )
    F.score(test, test_y, verbose=True)


if __name__ == '__main__':
    # Get data
    train, train_y = get_data_from_file('data/train.txt')
    test, test_y   = get_data_from_file('data/test.txt')
    # Run test
    test_model(LSTM, train, train_y, test, test_y)
    test_model(SparseLinearModel, train, train_y, test, test_y)
    test_model(TTBBTune, train, train_y, test, test_y, SENNA)
    test_model(TTBB, train, train_y, test, test_y, SENNA)
    test_model(fastTextPreTrain, train, train_y, test, test_y, SENNA)
    test_model(fastText, train, train_y, test, test_y)
    test_model(LinearModel, train, train_y, test, test_y, SENNA)
