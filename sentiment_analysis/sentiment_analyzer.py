from __future__ import division
from math import log, exp
import os
import csv

class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0


pos = MyDict()
neg = MyDict()
features = set()
totals = [0, 0]
delchars = ''.join(c for c in map(chr, range(128)) if not c.isalnum())



def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        # stripped = word.strip(delchars)
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = negated

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation

        if any(c in word for c in delims):
            negation = False

    return result


def train():
    global pos, neg, totals

    limit = 42700
    print('Training begins')
    root = './train/'
    for dir in os.listdir(root):
        for subDir in os.listdir(os.path.join(root, dir)):
            file_dir = os.path.join(root, dir, subDir)
            # print(file_dir)
            if subDir == 'pos':
                for file in os.listdir(file_dir)[:limit]:
                    for word in set(negate_sequence(open(file_dir+"/" + file).read())):
                        pos[word] += 1
                        pos['not_' + word] += 1
            if subDir == 'neg':
                for file in os.listdir(file_dir)[:limit]:
                    for word in set(negate_sequence(open(file_dir+"/" + file).read())):
                        neg[word] += 1
                        neg['not_' + word] += 1

    prune_features()

    totals[0] = sum(pos.values())
    totals[1] = sum(neg.values())

def classify(text):
    words = set(word for word in negate_sequence(text) if word in features)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    pos_prob = sum(log((pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob


def classify2(text):
    """
    For classification from pretrained data
    """
    words = set(word for word in negate_sequence(text) if word in pos or word in neg)
    if (len(words) == 0): return True
    # Probability that word occurs in pos documents
    pos_prob = sum(log((pos[word] + 1) / (2 * totals[0])) for word in words)
    neg_prob = sum(log((neg[word] + 1) / (2 * totals[1])) for word in words)
    return pos_prob > neg_prob


def classify_demo(text):
    words = set(word for word in negate_sequence(text) if word in pos or word in neg)
    if (len(words) == 0):
        # print "No features to compare on"
        return True

    pprob, nprob = 0, 0
    for word in words:
        pp = log((pos[word] + 1) / (2 * totals[0]))
        np = log((neg[word] + 1) / (2 * totals[1]))
        # print "%15s %.9f %.9f" % (word, exp(pp), exp(np))
        pprob += pp
        nprob += np

    # print ("Positive" if pprob > nprob else "Negative"), "log-diff = %.9f" % abs(pprob - nprob)


def MI(word):
    """
    Compute the weighted mutual information of a term.
    """
    T = totals[0] + totals[1]
    W = pos[word] + neg[word]
    I = 0
    if W == 0:
        return 0
    if neg[word] > 0:
        # doesn't occur in -ve
        I += (totals[1] - neg[word]) / T * log((totals[1] - neg[word]) * T / (T - W) / totals[1])
        # occurs in -ve
        I += neg[word] / T * log(neg[word] * T / W / totals[1])
    if pos[word] > 0:
        # doesn't occur in +ve
        I += (totals[0] - pos[word]) / T * log((totals[0] - pos[word]) * T / (T - W) / totals[0])
        # occurs in +ve
        I += pos[word] / T * log(pos[word] * T / W / totals[0])
    return I


def get_relevant_features():
    pos_dump = MyDict({k: pos[k] for k in pos if k in features})
    neg_dump = MyDict({k: neg[k] for k in neg if k in features})
    totals_dump = [sum(pos_dump.values()), sum(neg_dump.values())]
    return (pos_dump, neg_dump, totals_dump)


def prune_features():
    """
    Remove features that appear only once.
    """
    global pos, neg
    for k in pos.keys():
        if pos[k] <= 1 and neg[k] <= 1:
            del pos[k]

    for k in neg.keys():
        if neg[k] <= 1 and pos[k] <= 1:
            del neg[k]


def feature_selection_trials():
    """
    Select top k features. Vary k and plot data
    """
    global pos, neg, totals, features


    words = list(set(pos.keys() + neg.keys()))
    print ("Total no of features:", len(words))
    words.sort(key=lambda w: -MI(w))
    num_features, accuracy = [], []
    bestk = 0
    # limit = 24016
    limit = 6500
    path = "./test/"
    step = 5000
    start = 2000000
    best_accuracy = 0.0
    for w in words[:start]:
        features.add(w)
    for k in range(start, 3000000, step):
        for w in words[k:k + step]:
            features.add(w)
        correct = 0
        size = 0

        for file in os.listdir(path + "pos")[:limit]:
            correct += classify(open(path + "pos/" + file).read()) == True
            size += 1

        for file in os.listdir(path + "neg")[:limit]:
            correct += classify(open(path + "neg/" + file).read()) == False
            size += 1

        num_features.append(k + step)
        accuracy.append(correct / size)
        if (correct / size) > best_accuracy:
            bestk = k
        print(k + step, correct / size)

    features = set(words[:bestk])
    print (num_features, accuracy)


def test_classify():
    """
    Tests the dataset
    """
    total, correct = 0, 0
    for fname in os.listdir("./test/pos/"):
        correct += int(classify2(open("./test/pos/" + fname).read()) == True)
        total += 1
    for fname in os.listdir("./test/neg/"):
        correct += int(classify2(open("./test/neg/" + fname).read()) == False)
        total += 1
    print ("accuracy: %f" % (correct / total))






def findNegativeSentimentReviews():
    """
    Find the negative sentiment reviews
    """
    target = open('./data/negativeReviews.txt', 'w')
    with open('./data/comments.txt', 'r') as file:
        for line in file:
            if classify2(line.strip()) == False:
                print (line)
                target.write(line.strip())
                target.write('\n')



# Find Negative Reviews WRT Category
def findNegativeSentimentReviewsWRTCategory(value):
    """
    Find the negative sentiment reviews WRT to category
    """
    negativeReview = []
    for line in value:
        if classify2(line.strip()) == False:
            negativeReview.append(line)
    return negativeReview



# Find Negative Reviews Count
def findNegativeSentiReviewsCount(reviews):
    """
    Find the negative reviews count
    """
    count = 0
    for line in reviews:
        # print line
        if classify2(line.strip()) == False:
            count = count + 1
    return count



# classify WRT to category read file
def classifyAndWriteTofile(data):
    negativeReviewsWRTCategory = dict()
    for key, value in data.items():
        negativeReviewsWRTCategory[key] = findNegativeSentimentReviewsWRTCategory(value)
    count = 0
    for key, value in negativeReviewsWRTCategory.items():
        count += len(value)
    print (count)

    with open('./negativeReviewsWRTCategory.txt', 'w') as f:
        for key, value in negativeReviewsWRTCategory.items():
            f.write('%s:%s\n' % (key, value))




# Split the array with size attribute: len(array)/5
def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs



def readFileReviews():
    cleanComments = '../SentimentAnalysisUsingWord2V ec/Sentiment Analysis/cleanComments/'
    for root, dirs, files in os.walk(cleanComments):
        for name in files:
            fileName = name.split('.')[0]
            currentFile = os.path.join(root, name)
            print (fileName)

            with open(currentFile) as file:
                rows = file.readlines()
                if len(rows) > 5:
                    split_size = int(len(rows) / 5)
                    time_reviews = split(rows, split_size)
                    count = []
                    for reviews in time_reviews:
                        count.append((findNegativeSentiReviewsCount(reviews)/ len(reviews))* 100)
                    print (fileName, count[0],count[1], count[2], count[3], count[4])
                    csvwriter.writerow([fileName, count[0],count[1], count[2], count[3], count[4]])
                    file.close()
                else:
                    csvwriter.writerow([fileName, 0, 0, 0, 0, 0])
                    print (fileName, 0, 0, 0, 0, 0)


def readFileReviews_batch(batch_size, csvwriter):
    cleanComments = '../SentimentAnalysisUsingWord2V ec/Sentiment Analysis/cleanComments/'
    for root, dirs, files in os.walk(cleanComments):
        for name in files:
            fileName = name.split('.')[0]
            currentFile = os.path.join(root, name)
            # print fileName

            with open(currentFile) as file:
                rows = file.readlines()
                if len(rows) > batch_size:
                    # split_size = int(len(rows) / batch_size)
                    time_reviews = chunks(rows, batch_size)
                    # print(len(time_reviews))
                    count = []
                    for reviews in time_reviews:
                        # print (findNegativeSentiReviewsCount(reviews), len(rows))
                        count.append((findNegativeSentiReviewsCount(reviews) / len(rows)) * 100)
                    # print fileName,count
                    count = [fileName] + count
                    # print count
                    csvwriter.writerow(count)
                    file.close()
                else:
                    csvwriter.writerow([fileName])

import numpy as np
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return np.array_split(np.array(l),n)



# target_dir = './temporal_results/'

if __name__ == '__main__':
    # Train
    train()
    # Feature Selection
    feature_selection_trials()


    for batch_size in range(5,10,5):
        print ('batch_size: '+ str(batch_size))
        target = open('./temporal_results/SentimentComments'+str(batch_size)+'.csv', 'wb')
        csvwriter = csv.writer(target, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_columns = range(1,batch_size+1)
        csv_columns = [str('Time_') + str(s) for s in csv_columns]
        csv_columns = ['Id'] + csv_columns
        csvwriter.writerow(csv_columns)


    # data = dict()
    # data = readFromFile()
    # classifyAndWriteTofile(data)

    # data = dict()
    #     data = readFileReviews_batch(batch_size, csvwriter)
    #     target.close()

    # findNegativeSentimentReviews()
    # test_classify()
    # test_pang_lee()
    # classify_demo(open("pos_example").read())
    # classify_demo(open("neg_example").read())
