import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score


from dataset_builder import build_labeled_dataset

df = build_labeled_dataset()
X = df.drop(["target", "Close"], axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Binarize continuous features (threshold at median, or use your own domain thresholds)
X_train_bin = (X_train > X_train.median()).astype(int)
X_test_bin = (X_test > X_train.median()).astype(int)  # use train median to avoid leakage

nb_model = BernoulliNB()
nb_model.fit(X_train_bin, y_train)

y_pred = nb_model.predict(X_test_bin)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))