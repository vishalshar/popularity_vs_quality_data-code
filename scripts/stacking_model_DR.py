import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier


def tune_rf(x_train, y_train, x_test, y_test):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    train_results = []
    test_results = []
    for estimator in n_estimators:
       rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
       rf.fit(x_train, y_train)
       train_pred = rf.predict(x_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = rf.predict(x_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)
       from matplotlib.legend_handler import HandlerLine2D
       line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
       line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
       plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
       plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()

def get_600_data(fileName):
    data = pd.read_csv(fileName)

    # Y value
    Y = data['rating']
    data = data.drop(['rating'], axis = 1)

    # X value
    X = data
    return (X, Y)




print("loading data.....")


file_loc = '../SKfold/10/stage_3/'
# file_loc = '../SKfold/10/stage_2/'
# file_loc = '../SKfold/10/stage_1/'


'''
###########
# stage 3 #
###########
0.7492887624466573 0.6483143446706207
0.7405981490836805 0.9631666666666666 0.8352047864467501
0.869047619047619 0.33782051282051284 0.4614239028944912


###########
# stage 2 #
###########
0.7224593014066699 0.6146523776106891
0.7276990738065544 0.9344999999999999 0.8165494708040264
0.7048268398268399 0.3134615384615385 0.41275528441735193


###########
# stage 1 #
###########
0.7118579105421212 0.5908665673060969
0.7149900590922889 0.9426666666666665 0.8118626026673585
0.7190476190476189 0.2666666666666667 0.3698705319448354

'''



import os

precision_binary, recall_binary, f1_binary, accuracy = 0,0,0,0
precision_micro, recall_micro, f1_micro  = 0,0,0
precision_macro, recall_macro, f1_macro = 0,0,0
precision_weighted, recall_weighted, f1_weighted = 0,0,0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
n_comp = 2
random_state = 3

nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=n_comp,
                                                   random_state=random_state))


for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc+dirs)):
        print (root, dir, files)
        x_train, y_train = get_600_data(os.path.join(root+'/'+files[0]))
        x_test, y_test = get_600_data(os.path.join(root+'/'+files[1]))

        nca.fit(x_train, y_train)

        # clf3 = svm.SVC(kernel='rbf', decision_function_shape='ovo', degree=1, gamma='scale', coef0=1.0, shrinking=True, probability=True)
        clf1 = GradientBoostingClassifier(n_estimators=100, random_state=123)
        clf2 = RandomForestClassifier(n_estimators=1000, random_state=11)
        model = StackingClassifier(estimators=[('GBM', clf1), ('RF', clf2)], final_estimator=LogisticRegression(solver='liblinear'))

        # model.fit(x_train,y_train)
        model.fit(nca.transform(x_train), y_train)
        acc = model.score(nca.transform(x_test), y_test)

        print(acc)

        # predict = model.predict(x_test)
        #
        # import sklearn.metrics as metrics
        # print ("Accuracy: {}%".format(metrics.accuracy_score(y_test, predict)))
        # print ("Precision: {}%".format(100*metrics.precision_score(y_test, predict, average="weighted")))
        # print ("Recall: {}%".format(100*metrics.recall_score(y_test, predict, average="weighted")))
        # print ("f1_score: {}%".format(100*metrics.f1_score(y_test, predict, average="weighted")))
        # print (metrics.confusion_matrix(y_test, predict))
        #
        accuracy += acc

        # precision_0 += metrics.precision_score(y_test, predict, pos_label=0, average="binary")
        # recall_0 += metrics.recall_score(y_test, predict, pos_label=0, average="binary")
        # f1_0 += metrics.f1_score(y_test, predict, pos_label=0, average="binary")
        #
        # precision_1 += metrics.precision_score(y_test, predict, pos_label=1, average="binary")
        # recall_1 += metrics.recall_score(y_test, predict, pos_label=1, average="binary")
        # f1_1 += metrics.f1_score(y_test, predict, pos_label=1, average="binary")
        #
        # precision_binary += metrics.precision_score(y_test, predict, average="binary")
        # recall_binary += metrics.recall_score(y_test, predict, average="binary")
        # f1_binary += metrics.f1_score(y_test, predict, average="binary")
        #
        #
        # precision_micro += metrics.precision_score(y_test, predict, average="micro")
        # recall_micro += metrics.recall_score(y_test, predict, average="micro")
        # f1_micro += metrics.f1_score(y_test, predict, average="micro")
        #
        #
        # precision_macro += metrics.precision_score(y_test, predict, average="macro")
        # recall_macro += metrics.recall_score(y_test, predict, average="macro")
        # f1_macro += metrics.f1_score(y_test, predict, average="macro")
        #
        #
        # precision_weighted += metrics.precision_score(y_test, predict, average="weighted")
        # recall_weighted += metrics.recall_score(y_test, predict, average="weighted")
        # f1_weighted += metrics.f1_score(y_test, predict, average="weighted")



accuracy /= 10
print (accuracy)
#
# precision_0 /= 10
# recall_0 /= 10
# f1_0 /= 10
#
# precision_1 /= 10
# recall_1 /= 10
# f1_1 /= 10
#
# precision_binary /= 10
# recall_binary /= 10
# f1_binary /= 10
#
# precision_micro /= 10
# recall_micro /= 10
# f1_micro /= 10
#
# precision_macro /= 10
# recall_macro /= 10
# f1_macro /= 10
#
# precision_weighted /= 10
# recall_weighted /= 10
# f1_weighted /= 10


# print(accuracy, precision_binary, recall_binary, f1_binary)
# print(precision_0, recall_0, f1_0)
# print(precision_1, recall_1, f1_1)
# print( precision_micro, recall_micro, f1_micro)
# print( precision_macro, recall_macro, f1_macro)
# print( precision_weighted, recall_weighted, f1_weighted)

#
# print(accuracy, f1_macro)
# print(precision_1, recall_1, f1_1)
# print(precision_0, recall_0, f1_0)
