#import liberaries
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#handle categorical variable
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])
hot_encoder = OneHotEncoder(categorical_features = [3])
X = hot_encoder.fit_transform(X).toarray()

#avoid dummy var trap
X = X[:, 1:]

#spliting the dataset into test_set and trainig_test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#linear model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict
y_pred = regressor.predict(x_test)
