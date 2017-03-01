#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
sed -i -e '103186d' glove.840B.300d.txt
cd ..
echo "Processing data"
python -c "from shlo.utils import glove; glove()"
rm data/glove.840B.300d*
