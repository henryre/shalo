import cPickle
import numpy as np


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


def dump_data(fname, out_fname):
	data, labels = get_data_from_file(fname)
	with open(out_fname + '.data', 'wb') as f:
		cPickle.dump(data, f)
	with open(out_fname + '.labels', 'wb') as f:
		cPickle.dump(labels, f)


if __name__ == '__main__':

    dump_data('train.txt', 'polarity/train')
    dump_data('test.txt', 'polarity/test')
