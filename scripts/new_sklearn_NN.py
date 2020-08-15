import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc



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
0.7467322585743638 0.6784651902782745
0.7573367153178056 0.9106666666666665 0.8255720605140185
0.726969696969697 0.4314102564102564 0.5313583200425306

###########
# stage 2 #
###########
0.7228267741425636 0.6607890479090952
0.7530321275148861 0.8663333333333334 0.8047410427111142
0.6322438672438673 0.4461538461538462 0.5168370531070761

###########
# stage 1 #
###########
0.7011932985617196 0.633775886764085
0.7391954022988506 0.8496666666666668 0.7894074304570535
0.5828787878787878 0.41474358974358977 0.4781443430711166
'''



import os

precision_binary, recall_binary, f1_binary, accuracy = 0,0,0,0
precision_micro, recall_micro, f1_micro  = 0,0,0
precision_macro, recall_macro, f1_macro = 0,0,0
precision_weighted, recall_weighted, f1_weighted = 0,0,0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0

from sklearn.neural_network import MLPClassifier

for dirs in os.listdir(file_loc):
    for root, dir, files in os.walk(os.path.join(file_loc+dirs)):
        print (root, dir, files)
        x_train, y_train = get_600_data(os.path.join(root+'/'+files[0]))
        x_test, y_test = get_600_data(os.path.join(root+'/'+files[1]))

        # model = RandomForestClassifier(n_estimators=1000, random_state=11)
        # model.fit(x_train ,y_train)
        # predict = model.predict(x_test)

        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (64, 64), random_state = 1)
        model.fit(x_train, y_train)
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
