# SHLO

Shallow baseline models for text.

## Implemented models

* Linear model over pretrained embeddings
* fastText
* fastText with pretrained embeddings
* Simple but tough-to-beat-baseline for sentence embeddings (TTBB)
* TTBB with tuned embeddings and gradient-tuned common component
* LSTM baseline
* BOW one-hot baseline

## TODO model implementations

* TTBB with rank-one exact update
* TTBB with lazily updated common component
* Sequence models

## Getting started

### Install dependencies

* numpy
* tensorflow>=1.0

### Get data

```
data/get_senna.sh
data/get_polarity.sh
```

### Run tests for simple models

```
python shlo_test.py
```
