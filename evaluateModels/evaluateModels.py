import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("../resources/embeddings/faces.csv")

print(df.head())

# FEATURES
# X = np.array(df.drop("target", axis=1))
X = np.array(df.drop(df.columns[df.columns.str.contains('unnamed', case = False) | df.columns.str.contains('target', case = False)], axis = 1))

print(X[0])
print(X.shape)
