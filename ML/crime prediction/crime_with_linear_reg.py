# importing the liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("crime_data.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# spliting the dataset into training_set and test_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# fitting the linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Population vs Crimes Number (Training set)')
plt.xlabel('Population')
plt.ylabel('Crimes Number')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Population vs Crimes Number (Test set)')
plt.xlabel('Population')
plt.ylabel('Crimes Number')
plt.show()









