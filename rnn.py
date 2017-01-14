import tensorflow as tf
import cPickle as pickle
from collections import defaultdict
import re

#Read data and do preprocessing
def read_data(fn):
    with open(fn) as f:
        data = pickle.load(f)
    
    #Clean the text
    data_x, data_y = [],[]
    pattern = re.compile('[\W_]+')
    for text,label in data:
        text = text.strip("\r\n ").split()
        x = []
        for word in text:
            word = pattern.sub('', word)
            word = word.lower()
            if 0 < len(word) < 30:
                x.append(word)
        data_x.append(x)
        data_y.append(int(label))
    return data_x, data_y

train_x, train_y = read_data("data/train.p")
print train_x[0:10]

#build vocabulary from train_x
def build_vocab(data_x, min_count=100):
    counts = defaultdict(int)
    for x in data_x:
        for w in x:
            counts[w] += 1
    vocab = defaultdict(int)
    vocab["<unk>"] = 0
    for w,c in counts.iteritems():
        if c > min_count:
            vocab[w] = len(vocab)
    return vocab

def map_data(data_x, vocab):
    return [[vocab[w] for w in x] for x in data_x]

vocab = build_vocab(train_x)
vocab_size = len(vocab)
print "Vocab size:", len(vocab)

train_x = map_data(train_x, vocab)

import math

#build RNN model
batch_size = 20
hidden_size = 100
vocab_size = len(vocab)

def lookup_table(input_, vocab_size, output_size, name):
    with tf.variable_scope(name):
        embedding = tf.get_variable("embedding", [vocab_size, output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(output_size)))
    return tf.nn.embedding_lookup(embedding, input_)

def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b

session = tf.Session()

sentence = tf.placeholder(tf.int32, [batch_size, None])
label = tf.placeholder(tf.float32, [batch_size])

embedding = lookup_table(sentence, vocab_size, hidden_size, name="word_embedding")
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
init_state = lstm_cell.zero_state(batch_size, tf.float32)
_, final_state = tf.nn.dynamic_rnn(lstm_cell, embedding, initial_state=init_state)
sentiment = linear(final_state[1], 1, name="output")

sentiment = tf.squeeze(sentiment, [1])
loss = tf.nn.sigmoid_cross_entropy_with_logits(sentiment, label)

optimizer = tf.train.AdamOptimizer().minimize(loss / batch_size)

tf.global_variables_initializer().run(session=session)

