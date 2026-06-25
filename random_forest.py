from sklearn.metrics import classification_report
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from dataset_builder import  build_labeled_dataset, filter_dataset
from feature_engineering import build_features, FEATURES
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


labeled = build_labeled_dataset()
df = filter_dataset(labeled, FEATURES)
df = df.reset_index()
print(df.head())
X = df.drop(columns=["Datetime", "Close", "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
forest = RandomForestClassifier()


param_grid = {'n_estimators':[4, 8, 32, 64, 128],
              'max_features':[4, 6, 8, 10, 12, 'sqrt', 'log2'],
              'bootstrap': [True, False],
              'oob_score': [True, False]}

grid_model = GridSearchCV(estimator=forest,
                          param_grid=param_grid)
grid_model.fit(X_train, y_train)
y_pred = grid_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Out-of-bag score: {grid_model.best_estimator_.oob_score_}")