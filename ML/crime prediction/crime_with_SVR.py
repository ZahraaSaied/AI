# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('crime_data.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
y_sc = StandardScaler()
X = x_sc.fit_transform(X)
Y = y_sc.fit_transform(Y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

# Predicting a new result
#y_pred = y_sc.inverse_transform(regressor.predict(x_sc.transform(pred_arr)))

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Population vs Crimes Number')
plt.xlabel('Population')
plt.ylabel('Crimes Number')
plt.show()