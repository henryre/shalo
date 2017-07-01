#!/bin/bash

mkdir -p data
cd data
echo "Downloading Movie Review data"
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzvf aclImdb_v1.tar.gz