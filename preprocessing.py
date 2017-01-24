import tensorflow as tf
import cPickle as p
from collections import defaultdict
import re, random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import word2vec

trainData = p.load( open("data/train.p", 'rb'))
devData = p.load( open("data/dev.p", 'rb'))
testData = p.load( open("data/test.p", 'rb'))

w2vM = word2vec.load('data/emeddings.bin')
vocab = w2vM.vocab
vocabIndxDict = {}
for i,v in enumerate(vocab):
    vocabIndxDict[v] = i

raw_input("vocab")

UNK_ID = vocabIndxDict["unktoken"] 
PAD_ID = vocabIndxDict["padtoken"] 

def tweet2id(tweet):
    resTweet = []
    lenTweet = len(tweet)
    for i in range(20):
        if i < lenTweet:
            if tweet[i] in vocabIndxDict: 
                resTweet.append(vocabIndxDict[tweet[i]])
            else:
                resTweet.append(UNK_ID)
        else:
            resTweet.append(PAD_ID)
    return np.array(resTweet, dtype=np.int32)

train_data_pre = [ (tweet2id(tweet),label) for tweet,label in trainData]
dev_data_pre = [ (tweet2id(tweet),label) for tweet,label in devData]
test_data_pre = [ (tweet2id(tweet),label) for tweet,label in testData]

p.dump(train_data_pre, open('data/trainTweets_preprocessed.p','wb'))
p.dump(dev_data_pre, open('data/devTweets_preprocessed.p','wb'))
p.dump(test_data_pre, open('data/testTweets_preprocessed.p','wb'))
