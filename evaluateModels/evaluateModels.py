import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


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
    plt.savefig("../resources/models/confusion_matrix_{}.png".format(model_name))


df = pd.read_csv("../resources/embeddings/faces.csv")

print(df.head())

##########################################################
# FEATURES
##########################################################
# X = np.array(df.drop("target", axis=1))
X = np.array(df.drop(df.columns[df.columns.str.contains('unnamed', case = False) | df.columns.str.contains('target', case = False)], axis = 1))

print(X[0])
print(X.shape)

y = np.array(df.target)

print(y)

##########################################################
# SHUFFLE TRAIN DATA
##########################################################
trainX, trainY = shuffle(X, y, random_state=42)

##########################################################
# GET LABELS
##########################################################
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
print(trainY)

##########################################################
# VALIDATION
##########################################################
df_val = pd.read_csv("../resources/embeddings/faces_validation.csv")
print(df_val.head())

valX = np.array(df_val.drop(df_val.columns[df_val.columns.str.contains('unnamed', case = False) | df_val.columns.str.contains('target', case = False)], axis = 1))
print(valX)
print(valX.shape)
valY = np.array(df_val.target)
print(valY)
print(valY.shape)

out_encoder.fit(valY)
valY = out_encoder.transform(valY)
print(valY)

##########################################################
# Evaluate Models
##########################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(trainX, trainY)

yhat_train = knn.predict(trainX)
yhat_val = knn.predict(valX)

print(yhat_val)
print_confusion_matrix("KNN", valY, yhat_val)
