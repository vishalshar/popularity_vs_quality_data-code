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
0.7519954164691007 0.6970488709285702
0.7812196811860328 0.8778333333333335 0.8236010202551555
0.6907467532467533 0.5096153846153847 0.5704967216019848


###########
# stage 2 #
###########
0.7332780148569622 0.6857152084259631
0.779568055736534 0.8378333333333334 0.8054632275142645
0.628641774891775 0.5307692307692308 0.565967189337662


###########
# stage 1 #
###########
0.701264422317054 0.6368757416349147
0.7422416355431062 0.8416666666666668 0.786763812330617
0.5974242424242423 0.4301282051282051 0.48698767093921214

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc+dirs)):
        print (root, dir, files)
        x_train, y_train = get_600_data(os.path.join(root+'/'+files[0]))
        x_test, y_test = get_600_data(os.path.join(root+'/'+files[1]))


        # clf1 = svm.SVC(kernel='rbf', decision_function_shape='ovo', degree=2, gamma='scale', coef0=1.0, shrinking=True, probability=True)
        clf1 = GradientBoostingClassifier(n_estimators=100, random_state=123)
        clf2 = RandomForestClassifier(n_estimators=1000, random_state=11)
        # clf2 = RandomForestClassifier(n_estimators=100, random_state=11)
        model = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2)], voting='soft', )

        model.fit(x_train,y_train)
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
