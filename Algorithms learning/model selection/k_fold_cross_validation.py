# K - Fold cross validation

# Import the liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Split the dataset to training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)

# Fitting the classifier (kernel svm) model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# Applying K-Flod cross validation for model evaluating
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
average_accuracy = accuracies.mean()
variance = accuracies.std()

# Predict test set results
y_pred = classifier.predict(x_test)

# the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)