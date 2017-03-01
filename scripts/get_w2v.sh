#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2
bunzip2 deps.words.bz2
cd ..
echo "Processing data"
python -c "from shlo.data_utils import dep_w2v; dep_w2v()"
rm data/deps.words*
