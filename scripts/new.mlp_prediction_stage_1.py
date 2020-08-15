import numpy as np
import tensorflow as tf
import tflearn
import tflearn.layers.merge_ops
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import pickle as cPickle
import tflearn.datasets.mnist as mnist
import os
import scipy.io.wavfile
import random
import pandas as pd


def get_600_data(fileName):
    data = pd.read_csv(fileName)

    # print data
    # Replace ratings
    data['rating'][data['rating'] < 4.0] = 0
    data['rating'][data['rating'] >= 4.0] = 1

    # Y value
    Y = data['rating']
    data = data.drop(['rating'], axis=1)

    # Y = Y.replace(0, 5)
    # Y = Y.replace(1, 0)
    # Y = Y.replace(5, 1)

    # X value
    X = data

    return (X, Y)


def get_data(fileName):
    data = pd.read_csv(fileName)

    # Y value
    Y = data['rating']
    data = data.drop(['rating'], axis=1)

    # X value
    X = data
    return (X, Y)


print("loading data.....")
# fileLoc = './stage3.csv'
# fileLoc = './stage2.csv'
# fileLoc = './stage1.csv'
# fileLoc = './merge_data_600.csv'

# file_loc = './dataset/10/stage_3/'
# file_loc = './dataset/10/stage_2/'
file_loc = './dataset/10/stage_1/'


'''
0.6840707964601769 0.782443530736866 0.7193229590641349 0.7467098646733612
0.5466038509594509 0.6240447219849545 0.5771045013520235
0.782443530736866 0.7193229590641349 0.7467098646733612
0.6840707964601769 0.782443530736866 0.7193229590641349 0.7467098646733612
0.6840707964601769 0.6840707964601769 0.6840707964601769
0.6645236908481584 0.6716838405245447 0.6619071830126926
0.702358108871716 0.6840707964601769 0.6877503319183108'''

precision_binary, recall_binary, f1_binary, accuracy = 0, 0, 0, 0
precision_micro, recall_micro, f1_micro = 0, 0, 0
precision_macro, recall_macro, f1_macro = 0, 0, 0
precision_weighted, recall_weighted, f1_weighted = 0, 0, 0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0

# length = 20000
drop_out_prob = 0.25


def build_tflearn_ann(length):
    input_layer = input_data(shape=[None, length, 1])

    fc_layer_4 = fully_connected(input_layer, 1024, activation='relu', name='fc_layer_4', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_2 = dropout(fc_layer_4, drop_out_prob)
    batch_1 = batch_normalization(drop_2)

    fc_layer_5 = fully_connected(batch_1, 1024, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_3 = dropout(fc_layer_5, drop_out_prob)
    batch_2 = batch_normalization(drop_3)

    # merge_layer = tflearn.merge_outputs([fc_layer_4, fc_layer_5])
    # drop_2 = dropout(merge_layer, drop_out_prob)

    fc_layer_6 = fully_connected(batch_2, 512, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_4 = dropout(fc_layer_6, drop_out_prob)

    # # Output
    fc_layer_2 = fully_connected(drop_4, 2, activation='softmax', name='output')
    network = regression(fc_layer_2, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.0005,
                         metric='Accuracy')
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='./results/tensorboard/')
    return model


ann_model_dir = './model/ANN/'

##############
# Train ANN ##
##############

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale

NUM_EPOCHS = 100
BATCH_SIZE = 32

for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc + dirs)):
        print(root, dir, files)

        x_train, y_train = get_data(os.path.join(root + '/' + files[0]))
        x_test, y_test = get_data(os.path.join(root + '/' + files[1]))

        print("loading data")

        x_train = scale(x_train)
        x_test = scale(x_test)

        x_train = np.array(x_train)
        x_train = x_train[..., np.newaxis]

        x_test = np.array(x_test)
        x_test = x_test[..., np.newaxis]

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

        tf.reset_default_graph()
        MODEL = build_tflearn_ann(x_train.shape[1])
        MODEL.fit(x_train, np.array(y_train), n_epoch=NUM_EPOCHS,
                  shuffle=True,
                  validation_set=(x_test, y_test),
                  show_metric=True,
                  snapshot_epoch=True,
                  # snapshot_step = 100,
                  batch_size=BATCH_SIZE,
                  run_id='stage1')
        MODEL.save(ann_model_dir + 'Kickstarter_MLP.tfl')

        predict = MODEL.predict(x_test)
        predict_argmax = np.argmax(predict, axis=1)
        y_arxmax = np.argmax(y_test, axis=1)

        import sklearn.metrics as metrics

        print("Accuracy: {}%".format(metrics.accuracy_score(y_arxmax, predict_argmax)))
        print("Precision: {}%".format(100 * metrics.precision_score(y_arxmax, predict_argmax, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(y_arxmax, predict_argmax, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(y_arxmax, predict_argmax, average="weighted")))
        print(metrics.confusion_matrix(y_arxmax, predict_argmax))

        accuracy += metrics.accuracy_score(y_arxmax, predict_argmax)

        precision_0 += metrics.precision_score(y_arxmax, predict_argmax, pos_label=0, average="binary")
        recall_0 += metrics.recall_score(y_arxmax, predict_argmax, pos_label=0, average="binary")
        f1_0 += metrics.f1_score(y_arxmax, predict_argmax, pos_label=0, average="binary")

        precision_1 += metrics.precision_score(y_arxmax, predict_argmax, pos_label=1, average="binary")
        recall_1 += metrics.recall_score(y_arxmax, predict_argmax, pos_label=1, average="binary")
        f1_1 += metrics.f1_score(y_arxmax, predict_argmax, pos_label=1, average="binary")

        precision_binary += metrics.precision_score(y_arxmax, predict_argmax, average="binary")
        recall_binary += metrics.recall_score(y_arxmax, predict_argmax, average="binary")
        f1_binary += metrics.f1_score(y_arxmax, predict_argmax, average="binary")

        precision_micro += metrics.precision_score(y_arxmax, predict_argmax, average="micro")
        recall_micro += metrics.recall_score(y_arxmax, predict_argmax, average="micro")
        f1_micro += metrics.f1_score(y_arxmax, predict_argmax, average="micro")

        precision_macro += metrics.precision_score(y_arxmax, predict_argmax, average="macro")
        recall_macro += metrics.recall_score(y_arxmax, predict_argmax, average="macro")
        f1_macro += metrics.f1_score(y_arxmax, predict_argmax, average="macro")

        precision_weighted += metrics.precision_score(y_arxmax, predict_argmax, average="weighted")
        recall_weighted += metrics.recall_score(y_arxmax, predict_argmax, average="weighted")
        f1_weighted += metrics.f1_score(y_arxmax, predict_argmax, average="weighted")

accuracy /= 10



precision_0 /= 10
recall_0 /= 10
f1_0 /= 10

precision_1 /= 10
recall_1 /= 10
f1_1 /= 10

precision_binary /= 10
recall_binary /= 10
f1_binary /= 10

precision_micro /= 10
recall_micro /= 10
f1_micro /= 10

precision_macro /= 10
recall_macro /= 10
f1_macro /= 10

precision_weighted /= 10
recall_weighted /= 10
f1_weighted /= 10

print(accuracy, precision_binary, recall_binary, f1_binary)
print(precision_0, recall_0, f1_0)
print(precision_1, recall_1, f1_1)
print(accuracy, precision_binary, recall_binary, f1_binary)
print(precision_micro, recall_micro, f1_micro)
print(precision_macro, recall_macro, f1_macro)
print(precision_weighted, recall_weighted, f1_weighted)

'''



'''
