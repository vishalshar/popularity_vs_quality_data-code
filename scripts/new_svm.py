import numpy as np
import pickle as cPickle
import os
import random
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import svm


def get_600_data(fileName):
    data = pd.read_csv(fileName)

    # Y value
    Y = data['rating']
    data = data.drop(['rating'], axis=1)

    # Y = Y.replace(0, 5)
    # Y = Y.replace(1, 0)
    # Y = Y.replace(5, 1)

    # X value
    X = data
    return (X, Y)


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_params_


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

0.7123755334281651 0.6169223278386217
0.7252270847159777 0.9111666666666667 0.8065313072049232
0.6892460317460316 0.3282051282051282 0.42731334847232005


###########
# stage 2 #
###########

0.711861861861862 0.6119921058140689
0.7242063104159878 0.9148333333333334 0.8073042541587065
0.656060606060606 0.31923076923076926 0.4166799574694312


###########
# stage 1 #
###########

0.6985577682946105 0.590215619208952
0.7165838936249688 0.9066666666666666 0.7987377399658916
0.616031746031746 0.2961538461538462 0.3816934984520124

'''

from sklearn.preprocessing import scale

import os

precision_binary, recall_binary, f1_binary, accuracy = 0, 0, 0, 0
precision_micro, recall_micro, f1_micro = 0, 0, 0
precision_macro, recall_macro, f1_macro = 0, 0, 0
precision_weighted, recall_weighted, f1_weighted = 0, 0, 0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0


for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc + dirs)):
        print(root, dir, files)
        x_train, y_train = get_600_data(os.path.join(root + '/' + files[0]))
        x_test, y_test = get_600_data(os.path.join(root + '/' + files[1]))

        x_train = scale(x_train)
        x_test = scale(x_test)

        model = svm.SVC(kernel='rbf', decision_function_shape='ovo', degree=1, gamma='scale', coef0=1.0, shrinking=True)
        # model = svm.LinearSVC()
        model.fit(x_train, y_train)
        predict = model.predict(x_test)

        import sklearn.metrics as metrics

        print("Accuracy: {}%".format(metrics.accuracy_score(y_test, predict)))
        print("Precision: {}%".format(100 * metrics.precision_score(y_test, predict, average="weighted")))
        print("Recall: {}%".format(100 * metrics.recall_score(y_test, predict, average="weighted")))
        print("f1_score: {}%".format(100 * metrics.f1_score(y_test, predict, average="weighted")))
        print(metrics.confusion_matrix(y_test, predict))

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

print(accuracy, f1_macro)
# print (precision_binary, recall_binary, f1_binary)
print(precision_1, recall_1, f1_1)
print(precision_0, recall_0, f1_0)
# print(precision_micro, recall_micro, f1_micro)
# print(precision_macro, recall_macro, f1_macro)
# print(precision_weighted, recall_weighted, f1_weighted)

'''

'''
