#!/usr/bin/python

import os
import cPickle
import nltk

def process_imdb(fname, setting):
    labels, sentences = [], []
    dirname = fname + setting
    for i, root in enumerate([dirname+'/neg', dirname+'/pos']):
        label = i
        for filename in os.listdir(root): 
            with open(os.path.join(root, filename), 'rb') as f:
                for line in f:
                    text = nltk.word_tokenize(line.decode('utf-8'))
                    labels.append(int(label))
                    sentences.append(text)
    return sentences, labels

def dump_data(fname, data, labels):
    with open(fname + '.data', 'wb') as f:
        cPickle.dump(data, f)
    with open(fname + '.labels', 'wb') as f:
        cPickle.dump(labels, f)

if __name__ == '__main__':
    imdb_dir = 'data/aclImdb/'
    output_dir = 'data/imdb_data/'

    assert os.path.isdir('data'), "Please run from the shalo home directory"
    assert os.path.isdir(imdb_dir), "Please run get_movie_reviews.sh and then rerun this"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for setting in ['test', 'train']:
        print "Processing %s data"% setting
        sentences, labels = process_imdb(imdb_dir, setting)
        print "Dumping data"
        dump_data(output_dir+setting, sentences, labels)
