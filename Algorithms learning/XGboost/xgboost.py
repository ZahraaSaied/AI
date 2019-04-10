# XGBoost
# install : sudo python3 -m pip install xgboost
# install:  install -c conda-forge xgboost=0.6a2


# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
country_encoder = LabelEncoder()
X[:, 1] = country_encoder.fit_transform(X[:, 1])
gender_encoder = LabelEncoder()
X[:, 2] = gender_encoder.fit_transform(X[:, 2])
hot_encoder = OneHotEncoder(categorical_features = [1])
X = hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split the dataset int training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Apply XGboost calssifier
from xgboost import XGBClassiifier
classifier = XGBClassiifier()
classifier.fit(x_train, y_train)

# Applying K-Flod cross validation for model evaluating
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, njobs = -1)
average_accuracy = accuracies.mean()
variance = accuracies.std()


# Predict the test set result
y_pred = classifier.predict(x_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

