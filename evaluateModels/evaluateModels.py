import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from joblib import dump, load


def print_charts(model_name, history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig("../resources/reports/chart_acc_{}.png".format(model_name))
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig("../resources/reports/chart_loss_{}.png".format(model_name))
    plt.close()

def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("MODEL: {}".format(model_name))
    print("Accuracy: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))

    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    #plt.show()
    plt.savefig("../resources/reports/confusion_matrix_{}.png".format(model_name))
    plt.close()
#########################################################


##########################################################
# [TRAINING] GET EMBEDDINGS
##########################################################
df_knowns = pd.read_csv("../resources/embeddings/faces.csv")
df_unknowns = pd.read_csv("../resources/embeddings/faces_unknowns.csv")
df = pd.concat([df_unknowns, df_knowns])

print("Data Training Raw")
print(df.head())

##########################################################
# [TRAINING] FEATURES
##########################################################
# X = np.array(df.drop("target", axis=1))
X = np.array(df.drop(df.columns[df.columns.str.contains('unnamed', case = False) | df.columns.str.contains('target', case = False)], axis = 1))

print("Data Training Processed")
print(X[0])
print(X.shape)

y = np.array(df.target)

print(y)

##########################################################
# [TRAINING] SHUFFLE TRAIN DATA
##########################################################
#trainX, trainY = shuffle(X, y, random_state=42)

##########################################################
# [VALIDATION] PREPARE EMBEDDINGS FOR VALIDATION
##########################################################
df_val = pd.read_csv("../resources/embeddings/faces_validation.csv")
#df_val = pd.read_csv("../resources/embeddings/faces.csv")

print(df_val.head())

valX = np.array(df_val.drop(df_val.columns[df_val.columns.str.contains('unnamed', case = False) | df_val.columns.str.contains('target', case = False)], axis = 1))
print(valX)
print(valX.shape)
valY = np.array(df_val.target)
print(valY)
print(valY.shape)

##########################################################
# [VALIDATION] Split dataSets
##########################################################
from sklearn.model_selection import train_test_split
trainX, valX, trainY, valY = train_test_split(X, y, test_size=0.25, random_state=42)

##########################################################
# [VALIDATION] Normalize
##########################################################
from sklearn.preprocessing import Normalizer
norm = Normalizer(norm="l2")
trainX = norm.transform(trainX)
valX = norm.transform(valX)

##########################################################
# [VALIDATION] Process Labels
##########################################################
np.unique(trainY)
classes = len(np.unique(trainY))

##########################################################
# [VALIDATION] Target Labels encoder
##########################################################
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
print(trainY)

out_encoder.fit(valY)
valY = out_encoder.transform(valY)
print(valY)

##########################################################
# Evaluate Models
##########################################################

# SVM #############################################
#from sklearn import svm
#svm = svm.SVC()
#svm.fit(trainX, trainY)
#yhat_train = svm.predict(trainX)
#yhat_val = svm.predict(valX)
#print_confusion_matrix("SVM", valY, yhat_val)
#dump(svm, '../resources/models/faces_svm.h5')

# KNN #############################################
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=2)
#
#knn.fit(trainX, trainY)
#
#yhat_train = knn.predict(trainX)
#yhat_val = knn.predict(valX)
#
#print(yhat_val)
#print_confusion_matrix("KNN", valY, yhat_val)
#dump(knn, '../resources/models/faces_knn.h5')

# PERCEPTRON #############################################
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

# CONVERT LABELS TO CATEGORICAL
trainY = to_categorical(trainY)
valY = to_categorical(valY)

percep_model = models.Sequential()
percep_model.add(layers.Dense(128, activation="relu", input_shape=(128,)))
percep_model.add(layers.Dropout(0.5))

# 2 - KNOWN AND UNKNOWN
percep_model.add(layers.Dense(2, activation="softmax"))

percep_model.summary()
percep_model.compile(optimizer="adam", loss="categorical_crossentropy", #categorical_crossentropy or binary_crossentropy ?
              metrics=['accuracy'])

p_epochs = 40
p_batch_size = 8

history_model = percep_model.fit(trainX,
                 trainY,
                 epochs=p_epochs,
                 batch_size=p_batch_size,
                 validation_data=(valX, valY))

val_loss, val_acc = percep_model.evaluate(valX, valY)

yhat_train = percep_model.predict(trainX)
yhat_val = percep_model.predict(valX)

# CONVERT CATEGORICAL TO LABELS
yhat_val = np.argmax(yhat_val, axis=1)
valY = np.argmax(valY, axis=1)

print(yhat_val)

from datetime import datetime
strdatetime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S").upper()

print_confusion_matrix("PERCEPTRON_{}".format(strdatetime), valY, yhat_val)
print_charts("PERCEPTRON_{}".format(strdatetime), history_model)
percep_model.save("../resources/models/faces_percep_{}.h5".format(strdatetime))



