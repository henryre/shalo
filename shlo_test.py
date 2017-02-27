import numpy as np

from shlo import LinearModel, fastText


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
    F.train(train, train_y, seed=1701)
    print "[{0}] accuracy: {1:.2f}%".format(F.name, 100.*F.score(test, test_y))


if __name__ == '__main__':
    # Get data
    train, train_y = get_data_from_file('data/train.txt')
    test, test_y   = get_data_from_file('data/test.txt')
    # Run test
    test_model(LinearModel, train, train_y, test, test_y, SENNA)
    test_model(fastText, train, train_y, test, test_y)
