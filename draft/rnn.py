import tensorflow as tf
import cPickle as pickle
from collections import defaultdict
import re, random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#Read data and do preprocessing
def read_data(fn):
    with open(fn) as f:
        data = pickle.load(f)
    
    #Clean the text
    new_data = []
    pattern = re.compile('[\W_]+')
    for text,label in data:
        text = text.strip("\r\n ").split()
        x = []
        for word in text:
            word = pattern.sub('', word)
            word = word.lower()
            if 0 < len(word) < 20:
                x.append(word)
        new_data.append((' '.join(x),label))
    return new_data 

train = read_data("data/train.p")
print train[0:10]

train_x, train_y = zip(*train)
vectorizer = CountVectorizer(train_x, min_df=0.001) 
vectorizer.fit(train_x)
vocab = vectorizer.vocabulary_

UNK_ID = len(vocab)
PAD_ID = len(vocab) + 1
word2id = lambda w:vocab[w] if w in vocab else UNK_ID
train_x = [[word2id(w) for w in x.split()] for x in train_x]
train_data = zip(train_x, train_y)

import math

#build RNN model
batch_size = 20
hidden_size = 100
vocab_size = len(vocab) + 2

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

tweets = tf.placeholder(tf.int32, [batch_size, None])
labels = tf.placeholder(tf.float32, [batch_size])

embedding = lookup_table(tweets, vocab_size, hidden_size, name="word_embedding")
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
init_state = lstm_cell.zero_state(batch_size, tf.float32)
_, final_state = tf.nn.dynamic_rnn(lstm_cell, embedding, initial_state=init_state)
sentiment = linear(final_state[1], 1, name="output")

sentiment = tf.squeeze(sentiment, [1])
loss = tf.nn.sigmoid_cross_entropy_with_logits(sentiment, labels)
loss = tf.reduce_mean(loss)
prediction = tf.to_float(tf.greater_equal(sentiment, 0.5))
pred_err = tf.to_float(tf.not_equal(prediction, labels))
pred_err = tf.reduce_sum(pred_err)

optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.global_variables_initializer().run(session=session)
saver = tf.train.Saver()

random.shuffle(train_data)

err_rate = 0.0
for step in xrange(0, len(train_data), batch_size):
    batch = train_data[step:step+batch_size]
    batch_x, batch_y = zip(*batch)
    batch_x = list(batch_x)
    if len(batch_x) != batch_size:
        continue
    max_len = max([len(x) for x in batch_x])
    for i in xrange(batch_size):
        len_x = len(batch_x[i])
        batch_x[i] = [PAD_ID] * (max_len - len_x) + batch_x[i]
    batch_x = np.array(batch_x, dtype=np.int32)
    batch_y = np.array(batch_y, dtype=np.float32)
    feed_map = {tweets:batch_x, labels:batch_y}
    _, batch_err = session.run([optimizer, pred_err], feed_dict=feed_map)
    err_rate += batch_err
    if step % 100 == 0 and step > 0:
        print err_rate / step
    
