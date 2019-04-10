# Grid Search

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
classifier = SVC(kernel = 'rbf', gamma = 0.7, random_state = 0)
classifier.fit(x_train, y_train)

# Applying k-fold cross validation for model evaluation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
avrage_accuracy = accuracies.mean()
variance = accuracies.std()

#Grid search for choosing best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 10, 50, 100], 'kernel':['linear']},
              {'C':[1, 10, 50, 100], 'kernel':['rbf'], 'gamma':[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_prameters = grid_search.best_params_

# Predict test set results
y_pred = classifier.predict(x_test)

# the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

