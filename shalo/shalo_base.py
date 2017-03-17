import cPickle
import numpy as np
import tensorflow as tf

from time import time
from utils import LabelBalancer, symbol_embedding


SD = 0.1


class SHALOModel(object):
    """Base class for a shallow text classification model
    Trains a Tensorflow model with a linear layer over an arbitrary sentence
    featurization
    Child classes need to define
        @_preprocess_data: method for processing tokenized text
        @_embed_sentences: Tensorflow featurization of a sentence
    """

    name = 'SHALOModel'

    def __init__(self, save_file=None, n_threads=None):
        # Super constructor
        self.train_fn   = None
        self.loss       = None
        self.prediction = None
        self.save_dict  = None
        self.word_dict  = None
        self.session    = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=n_threads,
                inter_op_parallelism_threads=n_threads
            )
        ) if n_threads is not None else tf.Session()
        if save_file is not None:
            self.load(save_file)

    def _preprocess_data(self, sentence_data, init=True):
        """Data preprocessor
            @sentence_data: list of lists of words representing sentences
            @init: True if initializing (training) model, False for test
        """
        raise NotImplementedError()

    def _create_placeholders(self):
        """Define placeholders and return input, input_lengths, labels"""
        self.input         = tf.placeholder(
            tf.int32, [None, self.mx_len], name='input'
        )
        self.input_lengths = tf.placeholder(tf.int32, [None], name='input_lens')
        self.y             = tf.placeholder(tf.float32, [None], name='labels')

    def _embed_sentences(self):
        """Convert word features and sentence lengths to sentence embeddings
        Returns a tuple of a tensor with shape [batch_size x dim] representing 
        the sentence features and a dictionary of variables needed to save the
        graph
        """
        raise NotImplementedError()

    def _get_loss(self, logits, labels):
        """Return loss and prediction function"""
        if self.loss_function.lower() == 'log':
            return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels
            ))
        if self.loss_function.lower() == 'hinge':
            return tf.reduce_sum(tf.losses.hinge_loss(
                logits=logits, labels=labels
            ))
        raise Exception("Unknown loss <{0}>".format(self.loss))

    def _build(self):
        assert(self.d is not None)
        assert(self.lr is not None)
        assert(self.l2_penalty is not None)
        assert(self.loss_function is not None)
        # Get input placeholders and sentence features
        self._create_placeholders()
        sentence_feats, save_kwargs = self._embed_sentences()
        # Define linear model
        s1, s2 = self.seed, (self.seed + 1 if self.seed is not None else None)
        w = tf.Variable(tf.random_normal((self.d, 1), stddev=SD, seed=s1))
        b = tf.Variable(tf.random_normal((1, 1), stddev=SD, seed=s2))
        h = tf.squeeze(tf.matmul(sentence_feats, w) + b)
        # Define training procedure
        self.loss       = self._get_loss(h, self.y)
        self.loss      += self.l2_penalty * tf.nn.l2_loss(w)
        self.prediction = tf.sigmoid(h)
        self.train_fn   = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.save_dict  = save_kwargs.update({'w': w, 'b': b})

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
            x_batch_array[j, :t] = x[:t] if t > 0 else [self.word_dict.unknown]
            len_batch[j]         = max(t, 1)
        return x_batch_array, len_batch

    def _epoch_post_process(self, t):
        pass

    def _get_save_dict(self, **kwargs):
        # Will default to saving all global variables
        return None

    def train(self, sentence_data, sentence_labels, loss_function='log',
              n_epochs=20, lr=0.01, dim=50, batch_size=100, l2_penalty=0.0,
              ngrams=1, rebalance=False, max_sentence_length=None,
              dev_sentence_data=None, dev_labels=None, print_freq=5,
              seed=None, **kwargs):
        """Train SHALO model
            @sentence_data: list of lists of words representing sentences
            @sentence_labels: labels for sentences in [0, 1]
            @loss_function: loss function to use in ['log', 'hinge']
            @n_epochs: number of training epochs
            @lr: learning rate
            @dim: embedding dimension
            @batch_size: number of examples in each mini-batch
            @l2_penalty: L2 regularization strength
            @rebalance: rebalance training set?
            @max_sentence_length: if None, max in training set
            @dev_sentence_data: same as @sentence_data for dev set
            @dev_labels: same as @sentence_labels for dev set
            @print_freq: print stats after this many epochs
            @seed: random seed
            @kwargs: additional kwargs for child classes
        """
        self.train_kwargs = kwargs
        verbose = print_freq > 0
        if verbose:
            print("[{0}] dim={1} lr={2} l2={3} ngram={4}".format(
                self.name, dim, lr, l2_penalty, ngrams
            ))
            print("[{0}] Building model".format(self.name))
        # Get training data
        self.ngrams = ngrams
        train_data  = self._preprocess_data(sentence_data, init=True)
        self.mx_len = max_sentence_length or max(map(len, train_data))
        # Build model
        if hasattr(self, 'embeddings'):
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
            if dev_sentence_data is not None and dev_labels is not None:
                print("[{0}] Found dev set for training eval".format(self.name))
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
            self._epoch_post_process(t)
            # Print training stats
            if verbose and ((t+1) % print_freq == 0 or t in [0, (n_epochs-1)]):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAvg. loss = {3:.6f}"
                if dev_sentence_data is not None and dev_labels is not None:
                    acc = self.score(dev_sentence_data, dev_labels, 0.5, False)
                    msg += "\tDev acc. = {0:.2f}%".format(100. * acc)
                print(msg.format(self.name, t+1, time()-st, epoch_loss/n))
        if verbose:
            print("[{0}] Training done ({1:.2f}s)".format(self.name, time()-st))

    def predict(self, test_sentence_data, verbose=True):
        """Predict labels for test data"""
        test_data = self._preprocess_data(test_sentence_data, init=False)
        if verbose:
            is_unk = [w == self.word_dict.unknown for s in test_data for w in s]
            print "[{0}] Test set unknown token percentage: {1:.2f}%".format(
                self.name, 100. * float(sum(is_unk)) / len(is_unk)
            )
        test_array, test_len = self._get_data_batch(test_data)
        return np.ravel(self.session.run(
            self.prediction, self._get_feed(test_array, test_len, y_batch=None)
        ))

    def score(self, test_sentence_data, test_labels, b=0.5, verbose=True):
        """Score predictions on test data against gold labels"""
        yhat = self.predict(test_sentence_data, verbose)
        y    = np.ravel(test_labels)
        assert((yhat >= 0).all() and (yhat <= 1).all())
        assert((y >= 0).all() and (y <= 1).all())
        assert(len(y) == len(yhat))
        yhat_hard = yhat > b
        y_hard    = y > b
        acc = np.mean(yhat_hard == y_hard)
        if verbose:
            print "[{0}] Accuracy: {1:.2f}%".format(self.name, 100. * acc)
        return acc

    def save_info(self, model_name):
        sv = (
            self.d, self.lr, self.l2_penalty, self.mx_len, self.ngrams,
            self.train_kwargs
        )
        with open('{0}.info'.format(model_name), 'wb') as f:
            cPickle.dump(sv, f)

    def load_info(self, model_name):
        sv = (
            self.d, self.lr, self.l2_penalty, self.mx_len, self.ngrams,
            self.train_kwargs
        )
        with open('{0}.info'.format(model_name), 'rb') as f:
            sv = cPickle.load(f)

    def save(self, model_name=None, verbose=True):
        """Save current TensorFlow model
            @model_name: save file names
            @verbose: be talkative?
        """
        model_name = model_name or self.name
        self.save_info(model_name)
        save_dict = self.save_dict or tf.global_variables()
        saver = tf.train.Saver(save_dict)
        saver.save(self.session, './' + model_name, global_step=0)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, verbose=True):
        """Load TensorFlow model from file
            @model_name: save file names
            @verbose: be talkative?
        """
        self.load_info(model_name)
        self._build()
        load_dict = self.save_dict or tf.global_variables()
        saver = tf.train.Saver(load_dict)
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session, ckpt.model_checkpoint_path)
            if verbose:
                print("[{0}] Loaded model <{1}>".format(self.name, model_name))
        else:
            raise Exception("[{0}] No model found at <{1}>".format(
                self.name, model_name
            ))


class SHALOModelFixed(SHALOModel):

    name = 'SHALOModelFixed'

    def __init__(self, embedding_file, save_file=None, n_threads=None):
        SHALOModel.__init__(self, save_file, n_threads)
        with open(embedding_file, 'rb') as f:
            self.embedding_words, self.embeddings = cPickle.load(f)

    def _get_embedding(self):
        """
        Row 0 is 0 vector for no token
        Row 1 is 0 vector for unknown token
        Remaining rows are constant at pretrained emebdding
        """
        self.U = tf.constant(
            symbol_embedding(self.embeddings),
            dtype=tf.float32, name='embedding_matrix'
        )
        return self.U


class SHALOModelPreTrain(SHALOModel):

    name = 'SHALOModelPreTrain'

    def __init__(self, embedding_file, save_file=None, n_threads=None):
        SHALOModel.__init__(self, save_file, n_threads)
        with open(embedding_file, 'rb') as f:
            self.embedding_words, self.embeddings = cPickle.load(f)

    def _word_table_init(self, training_sentences):
        self._get_training_words(training_sentences)
        self.word_dict = SymbolTable()
        for word in self.embedding_words_train:
            self.word_dict.get(word)

    def _get_training_words(self, training_sentences):
        unique_words = set(w for s in training_sentences for w in s)
        training_embedding_idxs, self.embedding_words_train = [], []
        for i, word in enumerate(self.embedding_words):
            if word in unique_words:
                training_embedding_words.append(word)
                training_embedding_idxs.append(i)
        idxs = np.ravel(training_embedding_idxs)
        self.embeddings_train = self.embeddings[idxs, :]

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Row 1 is random initialization for UNKNOWN
        Rows 2 : 2 + len(self.embedding_words) are pretrained initialization
        Remaining rows are random initialization
        """
        zero = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed - 1
        unk = tf.Variable(tf.random_normal((1, self.d), stddev=SD, seed=s))
        pretrain = tf.Variable(self.embeddings_train, dtype=tf.float32)
        vecs = [zero, unk, pretrain]
        n_r = self.word_dict.num_words() - len(self.embedding_words_train)
        if n_r > 0:
            r = tf.Variable(tf.random_normal((n_r, self.d), stddev=SD, seed=s))
            vecs.append(r)
        self.U = tf.concat(vecs, axis=0, name='embedding_matrix')
        return self.U


class SHALOModelRandInit(SHALOModel):

    name = 'SHALOModelRandInit'

    def __init__(self, save_file=None, n_threads=None):
        SHALOModel.__init__(self, save_file, n_threads)

    def _get_embedding(self):
        """
        Return embedding tensor (either constant or variable)
        Row 0 is 0 vector for no token
        Initialize random embedding for UNKNOWN and all words
        """
        zero  = tf.constant(0.0, dtype=tf.float32, shape=(1, self.d))
        s = self.seed - 1
        embed = tf.Variable(tf.random_normal(
            (self.word_dict.num_words() + 1, self.d), stddev=SD, seed=s
        ))
        self.U = tf.concat([zero, embed], axis=0, name='embedding_matrix')
        return self.U
