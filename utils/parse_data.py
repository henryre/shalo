import numpy as np
import os

def get_data_from_file_polarity(fname):
    labels, sentences = [], []
    with open(fname, 'rb') as f:
        for line in f:
            label, text = line.strip().split(' ', 1)
            text = text.split(' ')
            labels.append((int(label) + 1) / 2)
            sentences.append(text)
    labels = np.ravel(labels)
    return sentences, labels

def get_data_from_file_imdb(fname):
    labels, sentences = [], []
    for i, root in enumerate([fname+'/neg', fname+'/pos']):
        label = i
        for filename in os.listdir(root): 
            with open(os.path.join(root, filename), 'rb') as f:
                for line in f:
                    # tokenize here
                    raise NotImplementedError("Work in progress. Tokenize data")
                    label, text = line.strip().split(' ', 1)
                    text = text.split(' ')
                    labels.append((int(label) + 1) / 2)
                    sentences.append(text)
            labels = np.ravel(labels)
    return sentences, labels