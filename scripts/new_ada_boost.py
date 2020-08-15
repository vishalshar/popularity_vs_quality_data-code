import numpy as np
import pickle as cPickle
import os
import random
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate



def get_600_data(fileName):
    data = pd.read_csv(fileName)

    # Y value
    Y = data['rating']
    data = data.drop(['rating'], axis = 1)

    # Y = Y.replace(0,5)
    # Y = Y.replace(1,0)
    # Y = Y.replace(5,1)

    # X value
    X = data
    return (X, Y)




print("loading data.....")

file_loc = '../SKfold/10/stage_3/'
# file_loc = '../SKfold/10/stage_2/'
# file_loc = '../SKfold/10/stage_1/'


'''

print(accuracy, f1_macro)
print(precision_1, recall_1, f1_1)
print(precision_0, recall_0, f1_0)

###########
# stage 3 #
###########
0.6984826932195353 0.6581353638604945
0.7834211369303358 0.7685 0.7702593587156655
0.5568804733278417 0.5634615384615385 0.5460113690053234


###########
# stage 2 #
###########
0.6796981191718033 0.6191932382857323
0.7455578361612846 0.7931666666666667 0.7630306488243322
0.5450500479912244 0.4602564102564103 0.47535582774713214


###########
# stage 1 #
###########
0.6799193930772878 0.6030729950027856
0.7370307878009075 0.8210000000000001 0.7703384073289182
0.5398717948717949 0.4064102564102563 0.43580758267665287

'''



import os

precision_binary, recall_binary, f1_binary, accuracy = 0,0,0,0
precision_micro, recall_micro, f1_micro  = 0,0,0
precision_macro, recall_macro, f1_macro = 0,0,0
precision_weighted, recall_weighted, f1_weighted = 0,0,0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0


for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc+dirs)):
        print (root, dir, files)
        x_train, y_train = get_600_data(os.path.join(root+'/'+files[0]))
        x_test, y_test = get_600_data(os.path.join(root+'/'+files[1]))

        model = AdaBoostClassifier(n_estimators=50, random_state=11)
        model.fit(x_train ,y_train)
        predict = model.predict(x_test)

        import sklearn.metrics as metrics
        print ("Accuracy: {}%".format(metrics.accuracy_score(y_test, predict)))
        print ("Precision: {}%".format(100*metrics.precision_score(y_test, predict, average="weighted")))
        print ("Recall: {}%".format(100*metrics.recall_score(y_test, predict, average="weighted")))
        print ("f1_score: {}%".format(100*metrics.f1_score(y_test, predict, average="weighted")))
        print (metrics.confusion_matrix(y_test, predict))

        accuracy += metrics.accuracy_score(y_test, predict)

        precision_0 += metrics.precision_score(y_test, predict, pos_label=0, average="binary")
        recall_0 += metrics.recall_score(y_test, predict, pos_label=0, average="binary")
        f1_0 += metrics.f1_score(y_test, predict, pos_label=0, average="binary")

        precision_1 += metrics.precision_score(y_test, predict, pos_label=1, average="binary")
        recall_1 += metrics.recall_score(y_test, predict, pos_label=1, average="binary")
        f1_1 += metrics.f1_score(y_test, predict, pos_label=1, average="binary")

        precision_binary += metrics.precision_score(y_test, predict, average="binary")
        recall_binary += metrics.recall_score(y_test, predict, average="binary")
        f1_binary += metrics.f1_score(y_test, predict, average="binary")


        precision_micro += metrics.precision_score(y_test, predict, average="micro")
        recall_micro += metrics.recall_score(y_test, predict, average="micro")
        f1_micro += metrics.f1_score(y_test, predict, average="micro")


        precision_macro += metrics.precision_score(y_test, predict, average="macro")
        recall_macro += metrics.recall_score(y_test, predict, average="macro")
        f1_macro += metrics.f1_score(y_test, predict, average="macro")


        precision_weighted += metrics.precision_score(y_test, predict, average="weighted")
        recall_weighted += metrics.recall_score(y_test, predict, average="weighted")
        f1_weighted += metrics.f1_score(y_test, predict, average="weighted")



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


# print(accuracy, precision_binary, recall_binary, f1_binary)
# print(precision_0, recall_0, f1_0)
# print(precision_1, recall_1, f1_1)
# print( precision_micro, recall_micro, f1_micro)
# print( precision_macro, recall_macro, f1_macro)
# print( precision_weighted, recall_weighted, f1_weighted)
print(accuracy, f1_macro)
print(precision_1, recall_1, f1_1)
print(precision_0, recall_0, f1_0)



'''

'''
