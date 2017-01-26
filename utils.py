import numpy as np

def one_hot(raw_data, vocab_size):
    data = np.zeros((len(raw_data), 20, vocab_size))
    for tweet_index in range(len(raw_data)):
        tweet = raw_data[tweet_index]
        for word_index in range(20):
            word_id = tweet[word_index]
            data[tweet_index, word_index, word_id] = 1
    return data