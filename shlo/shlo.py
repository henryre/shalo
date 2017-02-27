import cPickle
import numpy as np
import tensorflow as tf

from time import time
from utils import LabelBalancer, scrub, SymbolTable, TFModel


class SHLOModel(TFModel):

    def __init__(self, embedding_file=None, save_file=None, name='SHLOModel',
                 n_threads=None):
        # Super constructor
        super(SHLOModel, self).__init__(save_file, name, n_threads)
        # Get embedding file
        if embedding_file is not None:
            with open(embedding_file, 'rb') as f:
                self.embedding_words, self.embeddings = cPickle.load(f)
        else:
            self.embedding_words, self.embeddings = None, None

    def _preprocess_data(self, sentence_data, init=True):
        raise NotImplementedError()

    def _create_placeholders(self):
        """Define placeholders and return input, input_lengths, labels"""
        self.input         = tf.placeholder(
            tf.int32, [None, self.mx_len], name='input'
        )
        self.input_lengths = tf.placeholder(tf.int32, [None], name='input_lens')
        self.y             = tf.placeholder(tf.float32, [None], name='labels')
        return self.input, self.input_lengths, self.y

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 must be constant 0 vector
        """
        raise NotImplementedError()

    def _embed_sentences(self, word_features, lengths):
        """
        Convert word features and sentence lengths to sentence embeddings
        word_features has shape [batch_size x max_len x dim]
        lengths has shape [batch_size]
        """
        raise NotImplementedError()

    def _get_loss(self, logits, labels):
        """Return loss and prediction function"""
        if self.loss_function.lower() == 'log':
            return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels
            )), tf.sigmoid(logits)
        if self.loss_function.lower() == 'hinge':
            return tf.reduce_sum(tf.losses.hinge_loss(
                logits=logits, labels=labels
            )), None
        raise Exception("Unknown loss <{0}>".format(self.loss))

    def _get_save_dict(self, **kwargs):
        return kwargs

    def _build(self):
        assert(self.d is not None)
        assert(self.lr is not None)
        assert(self.l2_penalty is not None)
        assert(self.loss_function is not None)
        # Get input placeholders
        self.input, self.input_lengths, self.y = self._create_placeholders()
        # Generate sentence features
        word_embeddings = self._get_embedding()
        word_feats      = tf.nn.embedding_lookup(word_embeddings, self.input)
        sentence_feats  = self._embed_sentences(word_feats, self.input_lengths)
        # Define linear model
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        w = tf.Variable(tf.random_normal((self.d, 1), stddev=0.1, seed=s1))
        b = tf.Variable(tf.random_normal((1, 1), stddev=0.1, seed=s2))
        h = tf.squeeze(tf.matmul(sentence_feats, w) + b)
        # Define training procedure
        self.loss, self.prediction = self._get_loss(h, self.y)
        self.loss += self.l2_penalty * tf.nn.l2_loss(w)
        self.train_fn = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        assert(self.loss is not None)
        assert(self.prediction is not None)
        assert(self.train_fn is not None)
        self.save_dict = self._get_save_dict(w=w, b=b)

    def _get_feed(self, x_batch, len_batch, y_batch=None):
        feed = {self.input: x_batch, self.input_lengths: len_batch}
        if y_batch is not None:
            feed[self.y] = y_batch
        return feed

    def _get_data_batch(self, x_batch):
        m = len(x_batch)
        x_batch_array = np.zeros((m, self.mx_len)).astype(int)
        len_batch = np.zeros(m).astype(int)
        for j, x in enumerate(x_batch):
            t = min(self.mx_len, len(x))
            x_batch_array[j, :t] = x[:t]
            len_batch[j]         = t
        return x_batch_array, len_batch

    def train(self, sentence_data, sentence_labels, loss_function='log',
              n_epochs=20, lr=0.01, dim=50, batch_size=100, l2_penalty=0.0,
              rebalance=False, max_sentence_length=None, print_freq=5,
              seed=None):
        verbose = print_freq > 0
        if verbose:
            print("[{0}] dim={1} lr={2} l2={3}".format(
                self.name, dim, lr, l2_penalty
            ))
            print("[{0}] Building model".format(self.name))
        # Get training data
        train_data = self._preprocess_data(sentence_data, init=True)
        self.mx_len = max_sentence_length or max(map(len, train_data))
        # Build model
        if self.embeddings is not None:
            assert(self.embeddings.shape[1] == dim)
        self.d             = dim
        self.lr            = lr
        self.l2_penalty    = l2_penalty
        self.seed          = seed
        self.loss_function = loss_function
        self._build()
        # Get training indices
        np.random.seed(self.seed)
        train_idxs = LabelBalancer(sentence_labels).get_train_idxs(rebalance)
        X_train = [train_data[i] for i in train_idxs]
        y_train = np.ravel(sentence_labels)[train_idxs]
        # Run mini-batch SGD
        n = len(X_train)
        batch_size = min(batch_size, n)
        if verbose:
            st = time()
            print("[{0}] Training model".format(self.name))
            print("[{0}] #examples={1}  #epochs={2}  batch size={3}".format(
                self.name, n, n_epochs, batch_size
            ))
        self.session.run(tf.global_variables_initializer())
        for t in xrange(n_epochs):
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                r = min(n-1, i+batch_size)
                x_batch_array, len_batch = self._get_data_batch(X_train[i:r])
                loss, _ = self.session.run(
                    [self.loss, self.train_fn],
                    self._get_feed(x_batch_array, len_batch, y_train[i:r])
                )
                epoch_loss += loss
            # Print training stats
            if verbose and ((t+1) % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAvg. loss={3:.6f}"
                print(msg.format(self.name, t+1, time()-st, epoch_loss/n))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def predict(self, test_sentence_data):
        test_data = self._preprocess_data(test_sentence_data, init=False)
        is_unknown = [w == 1 for s in test_data for w in s]
        print "[{0}] Test set unknown token percentage: {1:.2f}%".format(
            self.name, 100. * float(sum(is_unknown)) / len(is_unknown)
        )
        test_array, test_len = self._get_data_batch(test_data)
        return np.ravel(self.session.run(
            self.prediction, self._get_feed(test_array, test_len, y_batch=None)
        ))

    def score(self, test_sentence_data, test_labels, b=0.5):
        yhat = self.predict(test_sentence_data)
        y    = np.ravel(test_labels)
        assert((yhat >= 0).all() and (yhat <= 1).all())
        assert((y >= 0).all() and (y <= 1).all())
        assert(len(y) == len(yhat))
        yhat_hard = yhat > b
        y_hard    = y > b
        return np.mean(yhat_hard == y_hard)

    def save_info(self, model_name):
        with open('{0}.info'.format(model_name), 'wb') as f:
            cPickle.dump((self.d, self.lr, self.l2_penalty, self.mx_len), f)

    def load_info(self, model_name):
        with open('{0}.info'.format(model_name), 'rb') as f:
            self.d, self.lr, self.l2_penalty, self.mx_len = cPickle.load(f)


class LinearModel(SHLOModel):

    def __init__(self, embedding_file, save_file=None, name='LinearModel',
                 n_threads=None):
        assert(embedding_file is not None)
        super(LinearModel, self).__init__(
            embedding_file, save_file, name, n_threads
        )

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
            for word in self.embedding_words:
                self.word_dict.get(word)
        # Process data
        return [
            [self.word_dict.lookup(scrub(word.lower())) for word in sentence]
            for sentence in sentence_data
        ]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Row 1 is 0 vector for unknown token
        """
        return tf.constant(np.vstack([
            np.zeros((2, self.d)),
            self.embeddings
        ]), dtype=tf.float32, name='embedding_matrix')

    def _embed_sentences(self, word_features, lengths):
        """Mean of word vectors"""
        s = tf.reduce_sum(word_features, axis=1)
        return s / tf.to_float(tf.reshape(lengths, (-1, 1)))


class fastText(SHLOModel):

    def __init__(self, save_file=None, name='fastText', n_threads=None):
        super(fastText, self).__init__(None, save_file, name, n_threads)

    def _preprocess_data(self, sentence_data, init=True):
        # Initialize word table and populate with embeddings
        if init:
            self.word_dict = SymbolTable()
        # Process data
        retrieve_fn = self.word_dict.get if init else self.word_dict.lookup
        return [
            [retrieve_fn(scrub(word.lower())) for word in sentence]
            for sentence in sentence_data
        ]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Initialize random embedding for UNKNOWN and all words
        """
        zero  = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed-1
        embed = tf.Variable(tf.random_normal(
            (self.word_dict.num_words() + 1, self.d), stddev=0.1, seed=s
        ))
        return tf.concat([zero, embed], axis=0, name='embedding_matrix')

    def _embed_sentences(self, word_features, lengths):
        """Mean of word vectors"""
        s = tf.reduce_sum(word_features, axis=1)
        return s / tf.to_float(tf.reshape(lengths, (-1, 1)))
