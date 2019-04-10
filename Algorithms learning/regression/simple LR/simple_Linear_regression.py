# Simple Linear regression Model

#Importing the liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#spliting the dataset into test_set and trainig_set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#fiting linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set
y_pred = regressor.predict(x_test)

#visualising the training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#visuaiising the test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()