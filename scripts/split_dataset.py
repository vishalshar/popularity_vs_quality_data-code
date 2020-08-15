import errno
import os

import numpy as np
import pandas as pd
import math
import sys
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers
import pandas as pd
import numpy as np


def get_data(fileName):
    data = pd.read_csv(fileName)
    data = data.drop_duplicates()
    data['rating'][data['rating'] < 4.0] = 0
    data['rating'][data['rating'] >= 4.0] = 1

    Y = data['rating']
    data = data.drop(['rating'], axis = 1)

    return (data, Y)




split_size = 10
# output = './SKfold/'+str(split_size)+'/stage_1/'
# data = get_data('./data/stage1.csv')


# output = './SKfold/'+str(split_size)+'/stage_2/'
# data = get_data('./data/stage2.csv')


output = './SKfold/'+str(split_size)+'/stage_3/'
X, Y = get_data('../data/stage3.csv')

X = np.array(X)
Y = np.array(Y)


drop_rate = 0.45
l2_regularizer = 0.008
bias_regularizer = 0.008
num_classes = 2

def get_model(input_shape):
    model = keras.Sequential()
    model.add(layers.Dense(16, input_shape=input_shape, activation='relu', kernel_regularizer=l2(l2_regularizer), bias_regularizer=l2(bias_regularizer)))
    model.add(keras.layers.Dropout(drop_rate, noise_shape=None, seed=None))
    model.add(layers.Dense(32, activation='relu',kernel_regularizer=l2(l2_regularizer), bias_regularizer=l2(bias_regularizer)))
    # model.add(keras.layers.Dropout(drop_rate, noise_shape=None, seed=None))
    # model.add(layers.Dense(64, activation='relu',kernel_regularizer=l2(l2_regularizer), bias_regularizer=l2(bias_regularizer)))
    # model.add(keras.layers.Dropout(drop_rate, noise_shape=None, seed=None))
    # model.add(layers.Dense(64, activation='relu',kernel_regularizer=l2(l2_regularizer), bias_regularizer=l2(bias_regularizer)))
    # model.add(keras.layers.Dropout(drop_rate, noise_shape=None, seed=None))
    # model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_regularizer), bias_regularizer=l2(bias_regularizer)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
# from keras.callbacks import EarlyStopping, ModelCheckpoint


precision_binary, recall_binary, f1_binary, accuracy = 0,0,0,0
precision_micro, recall_micro, f1_micro  = 0,0,0
precision_macro, recall_macro, f1_macro = 0,0,0
precision_weighted, recall_weighted, f1_weighted = 0,0,0
precision_0, recall_0, f1_0 = 0, 0, 0
precision_1, recall_1, f1_1 = 0, 0, 0

for train_index, test_index in skf.split(X, Y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    x_train = preprocessing.scale(X_train)
    x_test = preprocessing.scale(X_test)

    y_test = keras.utils.to_categorical(y_test, 2)
    y_train = keras.utils.to_categorical(y_train, 2)

    shape_dimension = len(np.array(x_train)[0])  # number of columns
    input_shape = (shape_dimension,)
    print(f'Feature shape: {input_shape}')

    # es_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode='auto')

    # model_checkpoint = ModelCheckpoint('./my_dir', save_best_only=True, save_weights_only=True, monitor='val_accuracy')

    model = get_model(input_shape)
    adam = Adam(learning_rate=0.01)
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.005, nesterov=True),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        epochs=100,
                        shuffle=True,
                        batch_size=8,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        # callbacks=[es_callback]
                        )

    # Test the model after training
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')

    predict = model.predict(x_test)
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

# print(accuracy, precision_binary, recall_binary, f1_binary)
# print(precision_0, recall_0, f1_0)
# print(precision_1, recall_1, f1_1)
# print( precision_micro, recall_micro, f1_micro)
# print( precision_macro, recall_macro, f1_macro)
# print( precision_weighted, recall_weighted, f1_weighted)

print(accuracy, f1_macro)
print(precision_1, recall_1, f1_1)
print(precision_0, recall_0, f1_0)

# for i in range(split_size):
    # train, test = train_test_split(data, test_size=0.3)

    # Create Folder
    # local_output = output + str(i)+'/'
    # try:
    #     os.makedirs(local_output)
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise
    #
    # # Save Data
    # train.to_csv(local_output+"train.csv", index=False)
    # test.to_csv(local_output+"test.csv", index=False)

