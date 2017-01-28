import csv
import pandas as pd
import re
import special_tokens
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

MIN_WORD_COUNT = 10

def flatten(nested):
    return [x for sublist in nested for x in sublist]  

def load_sentiment_data_bow():
    sentences = []
    data = pd.read_csv("./data/sentiment-tweets.csv")
    sentences = list(data['text'])
    sentiment_labels = data['airline_sentiment']

    sentences_of_words = [split_into_words(sentence) for sentence in sentences]
    word_counts_initial = get_word_counts(sentences_of_words)
    sentences_of_words = filter_words_by_count(sentences_of_words, word_counts_initial, MIN_WORD_COUNT)
    word_counts = get_word_counts(sentences_of_words)
    all_words = word_counts.keys()
    vocab_size = len(all_words)
    index_to_word = {index:word for index, word in enumerate(all_words)}
    word_to_index = {word: index for index, word in enumerate(all_words)}
    n_sentences = len(sentences_of_words)

    classes = set(sentiment_labels)
    n_classes = len(classes)

    X = np.zeros(shape=(n_sentences, vocab_size))
    y = np.zeros(shape=(n_sentences, n_classes))

    label_to_index = {'negative': -1, 'neutral': 0, 'positive':1}
    for sentence_index in range(n_sentences):
        current_sentence = sentences_of_words[sentence_index]
        for current_word_position in range(len(current_sentence)):
            word = sentences_of_words[sentence_index][current_word_position]
            token_index = word_to_index[word]
            X[sentence_index][token_index] += 1
        sentiment_label = sentiment_labels[sentence_index]
        sentiment_label_index = label_to_index[sentiment_label]
        y[sentence_index][sentiment_label_index] = 1
    return X, y, index_to_word, sentences


def load_sentiment_data(max_len):
    sentences = []
    data = pd.read_csv("./data/sentiment-tweets.csv")
    sentences = list(data['text'])
    sentiment_labels = data['airline_sentiment']

    sentences_of_words = [split_into_words(sentence) for sentence in sentences]
    word_counts_initial = get_word_counts(sentences_of_words)
    sentences_of_words = filter_words_by_count(sentences_of_words, word_counts_initial, MIN_WORD_COUNT)
    word_counts = get_word_counts(sentences_of_words)
    all_words = word_counts.keys()
    vocab_size = len(all_words)
    index_to_word = {index:word for index, word in enumerate(all_words)}
    word_to_index = {word: index for index, word in enumerate(all_words)}
    n_sentences = len(sentences_of_words)

    classes = set(sentiment_labels)
    n_classes = len(classes)

    X = np.zeros(shape=(n_sentences, max_len, vocab_size), dtype='float32')
    y = np.zeros(shape=(n_sentences, n_classes), dtype='float32')

    label_to_index = {'negative': 0, 'neutral': 1, 'positive':2}
    for sentence_index in range(n_sentences):
        current_sentence = sentences_of_words[sentence_index]

        for current_word_position in range(min(max_len, len(current_sentence))):
            word = sentences_of_words[sentence_index][current_word_position]
            token_index = word_to_index[word]
            X[sentence_index][current_word_position][token_index] = 1
        sentiment_label = sentiment_labels[sentence_index]
        sentiment_label_index = label_to_index[sentiment_label]
        y[sentence_index][sentiment_label_index] = 1
    return X, y, index_to_word, sentences

def filter_words_by_count(sentences, word_counts, cutoff=5):
    new_sentences = []
    for s_i in range(len(sentences)):
        sentence = sentences[s_i]
        new_sentence = []
        for w_i in range(len(sentence)):
            word = sentence[w_i]
            new_word = word
            if word_counts[word] < cutoff:
                new_word = special_tokens._UNK
            new_sentence.append(new_word)
        new_sentences.append(new_sentence)
    return new_sentences

def get_sentence_length_stats(sentences_of_words):
    print(np.mean([len(sentence) for sentence in sentences_of_words]))

def get_word_counts(sentences_of_words):
    word_counts = {0:special_word for special_word in special_tokens._START_VOCAB}
    for sentence in sentences_of_words:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def split_into_words(sentence):
  """Basic word splitting"""
  _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w.lower() for w in words if w]

def split_data(X, y, train_split=0.8, dev_split=0.1, test_split=0.1, random=False):
    """Splits data"""
    num_examples = len(X)
    indices = range(X.shape[0])
    if random:
        random.seed(42)
        random.shuffle(indices)
    boundary = int(num_examples*train_split)
    training_idx, test_idx = indices[:boundary], indices[boundary:]
    X_train, X_test = X[training_idx,:], X[test_idx,:]
    y_train, y_test = y[training_idx,:], y[test_idx,:]

    return X_train, y_train, X_test, y_test


def get_random_minibatch_indices(n_examples, batch_size):
    indices = range(n_examples)
    random.shuffle(indices)
    num_batches = n_examples/batch_size
    minibatch_indices = np.zeros(shape=(num_batches, batch_size), dtype='int32')
    for b_i in range(num_batches):
        for ex_i in range(batch_size):
            minibatch_indices[b_i] = indices[b_i*batch_size:(b_i+1)*batch_size]
    return minibatch_indices


def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

def bow_to_dict(bow_row, index_to_word):
    words = {}
    for i in range(len(bow_row)):
        word_count = bow_row[i]
        if word_count > 0:
            word = index_to_word[i]
            words[word] = words.get(word, 0)+word_count
    return words

def label_to_desc(label):
    return ["negative", "neutral", "positive"][np.argmax(label)]


def classify_and_plot(data, labels, x, out, session):
    outputs = []
    for i in range(len(data)):
        x_input = [data[i]]
        feed_dict = {x: x_input}
        output = session.run([out], feed_dict=feed_dict)
        outputs.append(output[0])

    plt.figure()
    plt.axis([0, 10, 0, 10])

    for i in range(len(outputs)):
        x_input = data[i]
        # print(outputs[i].shape)
        # print(float(outputs[i]))
        decision = 1 if float(outputs[i]) > 0.5 else 0
        label = labels[i]
        # print('ec', int(decision), int(label))
        # print('ec2', decision, label)
        # print(int(decision) is int(label))
        m_text = 'g' if int(decision) == int(label) else 'r'
        m_text += '_' if label == 0 else '+'
        plt.plot(x_input[0], x_input[1], m_text, markersize=10)
        
    plt.show()

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))

    xx = np.arange(0, 100)/10.0
    yy = np.arange(0, 100)/10.0

    mesh = np.array([[i, j] for i in range(100) for j in range(100)])/10.0

    # here "model" is your model's prediction (classification) function
    Z = session.run([out], feed_dict={x: mesh})[0]
        
    # print(Z)
    Z = np.array(Z)
    Z += 0.5
    Z = Z.astype(int)
    # Z = session.run([out], feed_dict={x_in:}) model(np.c_[xx.ravel(), yy.ravel()]) 

    # Put the result into a color plot
    Z = Z.reshape((100, 100))
    plt.contourf(xx, yy, Z)
    plt.show()

    print('predictions', outputs)

def one_hot(i, end):
    v = np.zeros(end)
    v[i] = 1
    return v