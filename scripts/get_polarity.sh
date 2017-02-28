#!/bin/bash

mkdir -p data
cd data
echo "Downloading data"
wget https://raw.githubusercontent.com/henryre/fastmulticontext/master/data/train.txt
wget https://raw.githubusercontent.com/henryre/fastmulticontext/master/data/test.txt
