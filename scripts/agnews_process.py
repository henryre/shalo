#!/usr/bin/python

import os
import cPickle
import nltk
import csv

def process_imdb(fname, setting):
    labels, sentences = [], []
    filename = setting + ".csv"
    with open(os.path.join(fname, filename), 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader: 
            if line[0] not in ["2", "4"]: # only include sports and tech
                continue
            label = 0 if line[0] ==  "2" else 1
            sentence = " ".join([line[1], line[2]]).replace("\"", "")
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
    agnews_dir = 'data/ag_news/ag_news_csv/'
    output_dir = 'data/agnews_data/'

    assert os.path.isdir('data'), "Please run from the shalo home directory"
    assert os.path.isdir(agnews_dir), "Please run get_movie_reviews.sh and then rerun this"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for setting in ['test', 'train']:
        print "Processing %s data"% setting
        sentences, labels = process_imdb(agnews_dir, setting)
        print "Dumping data"
        dump_data(output_dir+setting, sentences, labels)
