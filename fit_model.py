import argparse
import cPickle
import json

from collections import OrderedDict
from shalo import *
from shalo.model_search import ListParameter, RandomSearch
from sklearn.utils import shuffle

MODELS = OrderedDict([
    ('sparselm', SparseLinearModel), ('lstm', LSTM),
    ('linearmodel', LinearModel), 
    ('fasttext', fastText), ('fasttextpretrain', fastTextPreTrain),
    ('ttbb', TTBB), ('ttbbtune', TTBBTune), ('ttbbtunelazy', TTBBTuneLazy),
])


def read_config(fname):
	with open(fname, 'rb') as f:
		return json.load(f)


def load_data(data_fname, labels_fname):
	with open(data_fname, 'rb') as f:
		data = cPickle.load(f)
	with open(labels_fname, 'rb') as f:
		labels = cPickle.load(f)
	return (data, labels)


def run(model, config, embedding=None, word_freq=None, n_threads=None):
	# Define model
	model_kwargs = {'n_threads': n_threads}
	if embedding is not None:
		model_kwargs['embedding_file'] = embedding
	if word_freq is not None:
		model_kwargs['word_freq_file'] = word_freq
	F = model(**model_kwargs)
	# Load data
	train_X, train_y = load_data(config['train_data'], config['train_labels'])

	# Grab dev set
	train_X, train_y = shuffle(train_X, train_y)
	dev_size = len(train_X) / 10
	dev_X = train_X[dev_size:]
	train_X = train_X[:dev_size]
	dev_y = train_y[dev_size:]
	train_y = train_y[:dev_size]

	# Define search
	parameters = [
		ListParameter(k, v) for k, v in config['search_parameters'].iteritems()
	]
	S = RandomSearch(F, train_X, train_y, parameters, n=config.get('n', 25))
	# Run search
	b = config.get('b', 0.5)
	fixed_parameters = config.get('fixed_parameters', {})
	df = S.fit(dev_X, dev_y, b=b, **fixed_parameters)
	print df.head(10)
	# Score on test set
	test_X, test_y = load_data(config['test_data'], config['test_labels'])
	F.score(test_X, test_y, b=b, verbose=True)


if __name__ == '__main__':
	# Get command line arguments
	parser = argparse.ArgumentParser(description='Fit model to data set')
	parser.add_argument('model', action='store', help='Model name')
	parser.add_argument('experiment_config', action='store', help='Config file')
	parser.add_argument('-embedding', action='store', help='Embedding file')
	parser.add_argument('-word_freq', action='store', help='Word freq file')
	parser.add_argument('-n_threads', action='store', type=int, help='#Threads')
	args = parser.parse_args()
	# Run experiment
	if args.model.lower() in MODELS:
		model = MODELS[args.model.lower()]
	else:
		raise Exception('No model <{0}>. Options are {1}.'.format(
			args.model.lower(), MODELS.keys()
		))
	config = read_config(args.experiment_config)
	run(model, config, args.embedding, args.word_freq, args.n_threads)
