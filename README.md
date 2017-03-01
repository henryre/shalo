# SHLO

Shallow baseline models for text.

## Implemented models

* Linear model over pretrained embeddings
* fastText
* fastText with pretrained embeddings
* Simple but tough-to-beat-baseline for sentence embeddings (TTBB)
* TTBB with tuned embeddings and gradient-tuned common component
* TTBB with lazily updated common component (bad)
* LSTM baseline
* BOW one-hot baseline

## TODO model implementations

* Sequence models

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
python shlo_test.py
```
