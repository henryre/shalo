# SHALO

*Shallow baseline models for text in Tensorflow*

[![Build Status](https://travis-ci.org/henryre/shalo.svg?branch=master)](https://travis-ci.org/henryre/shalo)

## Implemented models

### Baseline models

* `SparseLM`: Linear model trained over sparse bag-of-words representation
* `LSTM`: Long short-term memory model
* `LSTMPreTrain`: LSTM with pretrained embeddings

### Mean word vector models

* `LinearModel`: Linear model over fixed, pretrained embeddings
* `fastText`: Implementation of [fastText](https://github.com/facebookresearch/fastText) in Tensorflow
* `fastTextPreTrained`: fastText initialized with pretrained embeddings

### "Tough-to-Beat Baseline" models

* `TTBB`: Implementation of [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx) in Tensorflow
* `TTBBTune`: `TTBB` with gradient-tuned embeddings, common component, and smoothing parameter
* `TTBBTuneExact`: `TTBB` with gradient-tuned embeddings and smoothing parameter, with common component updated exactly
* `TTBBTuneLazy`: `TTBB` with gradient-tuned embeddings and smoothing parameter, with common component updated once per epoch (does not work)


## Getting started

### Install dependencies

```
pip install --requirement python-package-requirements.txt
```

### Get data

```

chmod +x scripts/get_senna.sh
scripts/get_senna.sh
chmod +x scripts/get_word_freq.sh
scripts/get_word_freq.sh
chmod +x scripts/get_polarity.sh
scripts/get_polarity.sh

```

### Run tests for simple models

```
python shalo_test.py
```
