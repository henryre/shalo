#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
cd ..
echo "Processing data"
python -c "from shlo.data_utils import glove; glove()"
rm data/glove.840B.300d*
