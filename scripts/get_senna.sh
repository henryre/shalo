#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget https://github.com/baojie/senna/raw/master/embeddings/embeddings.txt
wget https://github.com/baojie/senna/raw/master/hash/words.lst
cd ..
echo "Processing data"
python -c "from shlo.utils import senna; senna()"
rm data/embeddings.txt
rm data/words.lst
