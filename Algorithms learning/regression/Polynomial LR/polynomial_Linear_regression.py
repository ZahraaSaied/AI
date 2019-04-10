import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly, Y)
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly, Y)

#visualising linear model
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Simple Linear Model")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#visualising polynomial model
plt.scatter(X, Y, color = "red")
plt.plot(X, poly_lin_reg.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Polynomial Linear Model")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#prepare prediction array
pred_arr = np.array([[6.5]])

#predicting with linear model
lin_reg.predict(pred_arr)

#predict with polynomial model 
poly_lin_reg.predict(poly_reg.fit_transform(pred_arr))







