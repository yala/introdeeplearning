import pickle as p
import random

random.seed(1)

dataPath = "twitterSentiments.csv"

# The following pickle files will contain a list of (tweet, sentiment) pairs
trainSetPath = "train.p"
trainLargeSetPath = "trainLarge.p"
devSetPath   = "dev.p"
testSetPath  = "test.p"

# Read twietterSentiments.csv and return a list of (tweet, sentiment) pairs
def getTweetDict(path):
    with open(path, 'rb') as f:
        lines = f.readlines()[1:] #Start from 1 since first line is a legend
        splitLines = [l.split(',') for l in lines]
        #Legend is: ItemID,Sentiment,SentimentSource,SentimentText
        tweets, sentiments = [line[-1] for line in splitLines], [line[1] for line in splitLines]

        return zip(tweets, sentiments)



## This file takes the twitter sentiment anaylsis dataset 
## and builds the python arrays we will be using for this project. 
## We build a training, development and test set using the text and labels provided.

if __name__ == "__main__":

    tweetSentimentPairs = getTweetDict(dataPath)

    random.shuffle(tweetSentimentPairs)

    trainSize = 60000    
    devSize = 20000
    testSize = 20000

    devStartIndex = -(devSize + testSize)

    #Set 60000 tweets to be our training set
    trainPairs = tweetSentimentPairs[:trainSize]
    p.dump(trainPairs, open(trainSetPath, 'wb'))
    assert len(trainPairs) == 60000

    #Set 1.5 mil tweets to be our training set
    trainLargePairs = tweetSentimentPairs[:devStartIndex]
    p.dump(trainLargePairs, open(trainLargeSetPath, 'wb'))

    #Set 20000 tweets to be our dev set 
    devPairs   = tweetSentimentPairs[ devStartIndex:-testSize]
    p.dump(devPairs, open(devSetPath, 'wb'))
    assert len(devPairs) == 20000
    
    #Set aside last 20000 tweets to be our test set
    testPairs   = tweetSentimentPairs[ -testSize:]
    assert len(testPairs) == 20000
    p.dump(testPairs, open(testSetPath, 'wb'))



    
