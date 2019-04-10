# classify different cars orgins

# import the libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("cars.csv", sep = ';')

# explore the dataset
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(dataset.describe())
print(dataset.isnull().sum())
print((dataset[['Car', 
                'MPG',
                'Cylinders',
                'Displacement',
                'Horsepower',
                'Weight',
                'Acceleration',
                'Model',
                'Origin']] == 0).sum())

# matrix of features and target variable
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values

# handle categoorical variable (target)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# split the dataset int training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

# feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)

# apply classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, 
                                    criterion =  'entropy',
                                    random_state = 20)
classifier.fit(x_train, y_train)

# k-fold cross validation for model evaluation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = x_train, 
                             y = y_train,
                             scoring = 'accuracy',
                             cv = 10)
average_accuracy = accuracies.mean()
variance = accuracies.std()

# predict test set results
y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




