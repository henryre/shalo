#!/usr/bin/python

import os
import cPickle
import nltk
import csv

def process_imdb(fname, setting):
    labels, sentences = [], []
    filename = setting + ".csv"
    quota = [0,0]
    if setting == 'test':
        maxquota = 5000
    else:
        maxquota = 15000
    with open(os.path.join(fname, filename), 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader: 
            label = 0 if line[0] ==  "1" else 1
            quota[label] += 1
            if quota[label] > maxquota:
                continue
            sentence = line[2].replace("\"", "")
            text = nltk.word_tokenize(sentence.decode('utf-8'))
            labels.append(int(label))
            sentences.append(text)
    return sentences, labels

def dump_data(fname, data, labels):
    with open(fname + '.data', 'wb') as f:
        cPickle.dump(data, f)
    with open(fname + '.labels', 'wb') as f:
        cPickle.dump(labels, f)

if __name__ == '__main__':
    amazon_dir = 'data/amazon/amazon_review_polarity_csv/'
    output_dir = 'data/amazon_data/'

    assert os.path.isdir('data'), "Please run from the shalo home directory"
    assert os.path.isdir(amazon_dir), "Please run get_movie_reviews.sh and then rerun this"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for setting in ['test', 'train']:
        print "Processing %s data"% setting
        sentences, labels = process_imdb(amazon_dir, setting)
        print "Dumping data"
        dump_data(output_dir+setting, sentences, labels)
