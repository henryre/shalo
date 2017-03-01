import numpy as np
from utils.parse_data import *
import argparse


from shlo import (
    fastText, fastTextPreTrain, LinearModel, # Vector mean models
    TTBB, TTBBTune, TTBBTuneLazy,            # TTBB models
    LSTM, SparseLinearModel,                 # Baseline models
)


SENNA = 'data/senna.pkl'

data_files = {
    'polarity':[get_data_from_file_polarity, 'data/train.txt', 'data/test.txt'],
    'imdb':[get_data_from_file_imdb, 'data/aclImdb/train/', 'data/aclImdb/test/']
}


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
    # Parse args
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset', type=str, 
                    help='What dataset do you want to use [imdb, polarity]')
    args = parser.parse_args()

    if args.dataset not in data_files.keys():
        print "Dataset must be one of ... [imdb, polarity]"
        exit()

    # Get data
    train, train_y = data_files[args.dataset][0](data_files[args.dataset][1])
    test, test_y   = data_files[args.dataset][0](data_files[args.dataset][2])
    # Run test
    #test_model(TTBBTuneLazy, train, train_y, test, test_y, SENNA)
    test_model(TTBBTune, train, train_y, test, test_y, SENNA)
    # test_model(TTBB, train, train_y, test, test_y, SENNA)
    # test_model(fastTextPreTrain, train, train_y, test, test_y, SENNA)
    # test_model(fastText, train, train_y, test, test_y)
    # test_model(LinearModel, train, train_y, test, test_y, SENNA)
    # test_model(LSTM, train, train_y, test, test_y)
    # test_model(SparseLinearModel, train, train_y, test, test_y)
