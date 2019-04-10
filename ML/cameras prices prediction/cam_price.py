# predict different cameras prices depends on their features

# import the libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("Camera.csv")

# explore the dataset
print(dataset.describe())
print(dataset.isnull().sum())
print((dataset[['Max resolution', 
                'Low resolution',
                'Effective pixels',
                'Zoom wide (W)',
                'Zoom tele (T)',
                'Normal focus range',
                'Macro focus range',
                'Storage included',
                'Weight (inc. batteries)',
                'Dimensions',
                'Price']] == 0).sum())

# matrix of features and target variable
X =dataset.iloc[:, 2:12].values
y = dataset.iloc[:, 12].values

# handle missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 0, strategy = "median")
X[:, 0:2] = imputer.fit_transform(X[:, 0:2])
X[:, 2:4] = imputer.fit_transform(X[:, 2:4])
X[:, 4:6] = imputer.fit_transform(X[:, 4:6])
X[:, 6:8] = imputer.fit_transform(X[:, 6:8])
X[:, 8:10] = imputer.fit_transform(X[:, 8:10])

# split the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

# Feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)
y_sc = StandardScaler()
y_train = y_sc.fit_transform(y_train.reshape(-1,1))
y_train = y_train.reshape(-1,)

# regression model
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly', degree = 3)
regressor.fit(x_train, y_train)

# k-fold cross validation for model evaluating
from sklearn.model_selection import cross_val_score
r_squared = cross_val_score(estimator = regressor,
                            X = x_train,
                            y = y_train,
                            scoring = 'r2',
                            cv = 10,
                            n_jobs = -1)
average_r = r_squared.mean() * -1
variance = r_squared.std()

# predict the test set reasults
y_pred = y_sc.inverse_transform(regressor.predict(x_test))