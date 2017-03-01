#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget http://norvig.com/ngrams/count_1w.txt
cd ..
echo "Processing data"
python -c "from utils.language_data_utils import word_freq; word_freq()"
rm data/count_1w.txt
