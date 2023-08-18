# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import pickle
import random
import sys
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed
from tensorflow.keras.models import load_model, Sequential
import numpy as np

#utils
import random
import re
import colorama
colorama.init()
def print_green(*args, **kwargs):
    """Prints green text to terminal"""
    print(colorama.Fore.GREEN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_cyan(*args, **kwargs):
    """Prints cyan text to terminal"""
    print(colorama.Fore.CYAN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def print_red(*args, **kwargs):
    """Prints red text to terminal"""
    print(colorama.Fore.RED, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def sample_preds(preds, temperature=1.0):
    """
    Samples an unnormalized array of probabilities. Use temperature to
    flatten/amplify the probabilities.
    """
    preds = np.asarray(preds).astype(np.float64)
    # Add a tiny positive number to avoid invalid log(0)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def word_tokenize(text):
    """
    Basic word tokenizer based on the Penn Treebank tokenization script, but
    setup to handle multiple sentences. Newline aware, i.e. newlines are
    replaced with a specific token. You may want to consider using a more robust
    tokenizer as a preprocessing step, and using the --pristine-input flag.
    """
    regexes = [
        # Starting quotes
        (re.compile(r'(\s)"'), r'\1 " '),
        (re.compile(r'([ (\[{<])"'), r'\1 " '),
        # Punctuation
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'([;@#$%&])'), r' \1 '),
        (re.compile(r'([?!\.])'), r' \1 '),
        (re.compile(r"([^'])' "), r"\1 ' "),
        # Parens and brackets
        (re.compile(r'([\]\[\(\)\{\}\<\>])'), r' \1 '),
        # Double dashes
        (re.compile(r'--'), r' -- '),
        # Ending quotes
        (re.compile(r'"'), r' " '),
        (re.compile(r"([^' ])('s|'m|'d) "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'re|'ve|n't) "), r"\1 \2 "),
        # Contractions
        (re.compile(r"\b(can)(not)\b"), r' \1 \2 '),
        (re.compile(r"\b(d)('ye)\b"), r' \1 \2 '),
        (re.compile(r"\b(gim)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(gon)(na)\b"), r' \1 \2 '),
        (re.compile(r"\b(got)(ta)\b"), r' \1 \2 '),
        (re.compile(r"\b(lem)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(mor)('n)\b"), r' \1 \2 '),
        (re.compile(r"\b(wan)(na)\b"), r' \1 \2 '),
        # Newlines
        (re.compile(r'\n'), r' \\n ')
    ]

    text = " " + text + " "
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.split()


def word_detokenize(tokens):
    """
    A heuristic attempt to undo the Penn Treebank tokenization above. Pass the
    --pristine-output flag if no attempt at detokenizing is desired.
    """
    regexes = [
        # Newlines
        (re.compile(r'[ ]?\\n[ ]?'), r'\n'),
        # Contractions
        (re.compile(r"\b(can)\s(not)\b"), r'\1\2'),
        (re.compile(r"\b(d)\s('ye)\b"), r'\1\2'),
        (re.compile(r"\b(gim)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(gon)\s(na)\b"), r'\1\2'),
        (re.compile(r"\b(got)\s(ta)\b"), r'\1\2'),
        (re.compile(r"\b(lem)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(mor)\s('n)\b"), r'\1\2'),
        (re.compile(r"\b(wan)\s(na)\b"), r'\1\2'),
        # Ending quotes
        (re.compile(r"([^' ]) ('ll|'re|'ve|n't)\b"), r"\1\2"),
        (re.compile(r"([^' ]) ('s|'m|'d)\b"), r"\1\2"),
        (re.compile(r'[ ]?"'), r'"'),
        # Double dashes
        (re.compile(r'[ ]?--[ ]?'), r'--'),
        # Parens and brackets
        (re.compile(r'([\[\(\{\<]) '), r'\1'),
        (re.compile(r' ([\]\)\}\>])'), r'\1'),
        (re.compile(r'([\]\)\}\>]) ([:;,.])'), r'\1\2'),
        # Punctuation
        (re.compile(r"([^']) ' "), r"\1' "),
        (re.compile(r' ([?!\.])'), r'\1'),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
        (re.compile(r'([#$]) '), r'\1'),
        (re.compile(r' ([;%:,])'), r'\1'),
        # Starting quotes
        (re.compile(r'(")[ ]?'), r'"')
    ]

    text = ' '.join(tokens)
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.strip()


def find_random_seeds(text, num_seeds=50, max_seed_length=50):
    """Heuristic attempt to find some good seed strings in the input text"""
    lines = text.split('\n')
    # Take a random sampling of lines
    if len(lines) > num_seeds * 4:
        lines = random.sample(lines, num_seeds * 4)
    # Take the top quartile based on length so we get decent seed strings
    lines = sorted(lines, key=len, reverse=True)
    lines = lines[:num_seeds]
    # Split on the first whitespace before max_seed_length
    return [line[:max_seed_length].rsplit(None, 1)[0] for line in lines]


def shape_for_stateful_rnn(data, batch_size, seq_length, seq_step):
    """
    Reformat our data vector into input and target sequences to feed into our
    RNN. Tricky with stateful RNNs.
    """
    # Our target sequences are simply one timestep ahead of our input sequences.
    # e.g. with an input vector "wherefore"...
    # targets:   h e r e f o r e
    # predicts   ^ ^ ^ ^ ^ ^ ^ ^
    # inputs:    w h e r e f o r
    inputs = data[:-1]
    targets = data[1:]

    # We split our long vectors into semi-redundant seq_length sequences
    inputs = _create_sequences(inputs, seq_length, seq_step)
    targets = _create_sequences(targets, seq_length, seq_step)

    # Make sure our sequences line up across batches for stateful RNNs
    inputs = _batch_sort_for_stateful_rnn(inputs, batch_size)
    targets = _batch_sort_for_stateful_rnn(targets, batch_size)

    # Our target data needs an extra axis to work with the sparse categorical
    # crossentropy loss function
    targets = targets[:, :, np.newaxis]
    return inputs, targets

def _create_sequences(vector, seq_length, seq_step):
    # Take strips of our vector at seq_step intervals up to our seq_length
    # and cut those strips into seq_length sequences
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        num_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples,
                                 (num_pass_samples, seq_length))
        passes.append(pass_samples)
    # Stack our sequences together. This will technically leave a few "breaks"
    # in our sequence chain where we've looped over are entire dataset and
    # return to the start, but with large datasets this should be neglegable
    return np.concatenate(passes)

def _batch_sort_for_stateful_rnn(sequences, batch_size):
    # Now the tricky part, we need to reformat our data so the first
    # sequence in the nth batch picks up exactly where the first sequence
    # in the (n - 1)th batch left off, as the RNN cell state will not be
    # reset between batches in the stateful model.
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        # Take a slice of num_batches consecutive samples
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        # Spread it across each of our batches in the same index position
        reshuffled[batch_index::batch_size, :] = index_slice
    return reshuffled
    
    
from collections import Counter
class Vectorizer:
    """
    Transforms text to vectors of integer numbers representing in text tokens
    and back. Handles word and character level tokenization.
    """
    def __init__(self, text, word_tokens, pristine_input, pristine_output):
        self.word_tokens = word_tokens
        self._pristine_input = pristine_input
        self._pristine_output = pristine_output

        tokens = self._tokenize(text)
        print('corpus length:', len(tokens))
        token_counts = Counter(tokens)
        # Sort so most common tokens come first in our vocabulary
        tokens = [x[0] for x in token_counts.most_common()]
        self._token_indices = {x: i for i, x in enumerate(tokens)}
        self._indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('vocab size:', self.vocab_size)

    def _tokenize(self, text):
        if not self._pristine_input:
            text = text.lower()
        if self.word_tokens:
            if self._pristine_input:
                return text.split()
            return word_tokenize(text)
        return text

    def _detokenize(self, tokens):
        if self.word_tokens:
            if self._pristine_output:
                return ' '.join(tokens)
            return word_detokenize(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        """Transforms text to a vector of integers"""
        tokens = self._tokenize(text)
        indices = []
        for token in tokens:
            if token in self._token_indices:
                indices.append(self._token_indices[token])
            else:
                print_red('Ignoring unrecognized token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        """Transforms a vector of integers back to text"""
        tokens = [self._indices_token[index] for index in vector.tolist()]
        return self._detokenize(tokens)
        
        
        
class LiveSamplerCallback(Callback):
    """
    Live samples the model after each epoch, which can be very useful when
    tweaking parameters and/or the dataset.
    """
    def __init__(self, meta_model):
        super(LiveSamplerCallback, self).__init__()
        self.meta_model = meta_model

    def on_epoch_end(self, epoch, logs=None):
        print()
        print_green('Sampling model...')
        self.meta_model.update_sample_model_weights()
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('Using diversity:', diversity)
            self.meta_model.sample(diversity=diversity)
            print('-' * 50)


class MetaModel:
    """
    We wrap the keras model in our own metaclass that handles text loading,
    and provides convenient train and sample functions.
    """
    def __init__(self):
        self.train_model = None
        self.sample_model = None
        self.seeds = None
        self.vectorizer = None

    # Read in our data and validation texts
    def _load_data(self, data_dir, word_tokens, pristine_input, pristine_output,
                   batch_size, seq_length, seq_step):
        try:
        
            #with open(os.path.join(data_dir, 'input.txt')) as input_file:
            #    text = input_file.read()
            text = ""    
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt'):  # Nur Textdateien berücksichtigen
                    file_path = os.path.join(data_dir, filename)
                    with open(file_path, 'r') as file:
                        tmp_text = file.read()
                        text += tmp_text
                
        except FileNotFoundError:
            print_red("No input.txt in data_dir")
            sys.exit(1)

        skip_validate = True
        try:
            with open(os.path.join(data_dir, 'validate.txt')) as validate_file:
                text_val = validate_file.read()
                skip_validate = False
        except FileNotFoundError:
            pass # Validation text optional

        # Find some good default seed string in our source text.
        self.seeds = find_random_seeds(text)
        # Include our validation texts with our vectorizer
        all_text = text if skip_validate else '\n'.join([text, text_val])
        self.vectorizer = Vectorizer(all_text, word_tokens,
                                     pristine_input, pristine_output)

        data = self.vectorizer.vectorize(text)
        x, y = shape_for_stateful_rnn(data, batch_size, seq_length, seq_step)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

        if skip_validate:
            return x, y, None, None

        data_val = self.vectorizer.vectorize(text_val)
        x_val, y_val = shape_for_stateful_rnn(data_val, batch_size,
                                              seq_length, seq_step)
        print('x_val.shape:', x_val.shape)
        print('y_val.shape:', y_val.shape)
        return x, y, x_val, y_val

    # Builds the underlying keras model
    def _build_models(self, batch_size, embedding_size, rnn_size, num_layers):
        model = Sequential()
        model.add(Embedding(self.vectorizer.vocab_size,
                            embedding_size,
                            batch_input_shape=(batch_size, None)))
        for layer in range(num_layers):
            model.add(LSTM(rnn_size,
                           stateful=True,
                           return_sequences=True))
            model.add(Dropout(0.2))
        
        
        model.add(TimeDistributed(Dense(self.vectorizer.vocab_size, activation='softmax')))
        
        # Specify the learning rate
        learning_rate = 0.00005  # You can adjust this value
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        # With sparse_categorical_crossentropy we can leave as labels as
        # integers instead of one-hot vectors
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer= optimizer, #'rmsprop',
                      metrics=['accuracy'])
                      
        model.summary()

        # Keep a separate model with batch_size 1 for sampling
        self.train_model = model
        config = model.get_config()
        config['layers'][0]['config']['batch_input_shape'] = (1, None)
        self.sample_model = Sequential.from_config(config)
        self.sample_model.trainable = False

    def update_sample_model_weights(self):
        """Sync training and sampling model weights"""
        self.sample_model.set_weights(self.train_model.get_weights())

    def train(self, data_dir, word_tokens, pristine_input, pristine_output,
              batch_size, seq_length, seq_step, embedding_size, rnn_size,
              num_layers, num_epochs, live_sample):
        """Train the model"""
        print_green('Loading data...')
        load_start = time.time()
        x, y, x_val, y_val = self._load_data(data_dir, word_tokens,
                                             pristine_input, pristine_output,
                                             batch_size, seq_length, seq_step)
        load_end = time.time()
        print_red('Data load time', load_end - load_start)

        print_green('Building model...')
        model_start = time.time()
        self._build_models(batch_size, embedding_size, rnn_size, num_layers)
        model_end = time.time()
        print_red('Model build time', model_end - model_start)

        print_green('Training...')
        train_start = time.time()
        validation_data = (x_val, y_val) if (x_val is not None) else None
        callbacks = [LiveSamplerCallback(self)] if live_sample else None
        self.train_model.fit(x, y,
                             validation_data=validation_data,
                             batch_size=batch_size,
                             shuffle=False,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=callbacks)
        self.update_sample_model_weights()
        train_end = time.time()
        print_red('Training time', train_end - train_start)

    def sample(self, seed=None, length=None, diversity=1.0):
        """Sample the model"""
        self.sample_model.reset_states()

        if length is None:
            length = 100 if self.vectorizer.word_tokens else 500

        if seed is None:
            seed = random.choice(self.seeds)
            print('Using seed: ', end='')
            print_cyan(seed)
            print('-' * 50)

        preds = None
        seed_vector = self.vectorizer.vectorize(seed)
        # Feed in seed string
        print_cyan(seed, end=' ' if self.vectorizer.word_tokens else '')
        for char_index in np.nditer(seed_vector):
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)

        sampled_indices = np.array([], dtype=np.int32)
        # Sample the model one token at a time
        for i in range(length):
            char_index = 0
            if preds is not None:
                char_index = sample_preds(preds[0][0], diversity)
            sampled_indices = np.append(sampled_indices, char_index)
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)
        sample = self.vectorizer.unvectorize(sampled_indices)
        print(sample)
        return sample

    # Don't pickle the keras models, better to save directly
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['train_model']
        del state['sample_model']
        return state


def save(model, data_dir):
    """Save the keras model directly and pickle our meta model class"""
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model.sample_model.save(filepath=keras_file_path)
    pickle.dump(model, open(pickle_file_path, 'wb'))
    print_green('Model saved to', pickle_file_path, keras_file_path)


def load(data_dir):
    """Load the meta model and restore its internal keras model"""
    keras_file_path = os.path.join(data_dir, 'model.h5')
    pickle_file_path = os.path.join(data_dir, 'model.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.sample_model = load_model(keras_file_path)
    return model
