# Predicting number of videos views based on number of likes, dislikes and subscribers

# import the data
import pandas as pd 
X = pd.read_csv("StatsVideosXALL.csv")
y = pd.read_csv("StatsVideosYALL.csv")

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# build the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 200)
model.fit(x_train, y_train)

# predicting
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# evaluating
from sklearn.metrics import r2_score
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

# cross validation
from sklearn.model_selection import cross_val_score
r2_training = cross_val_score(estimator = model, 
                     X = x_train, 
                     y = y_train, 
                     cv = 10, 
                     scoring = 'r2')
average_r2_training = r2_training.mean()
std_training = r2_training.std()


r2_testing = cross_val_score(estimator = model, 
                     X = x_test, 
                     y = y_test, 
                     cv = 10, 
                     scoring = 'r2')
average_r2_testing = r2_testing.mean()
std_testing = r2_testing.std()