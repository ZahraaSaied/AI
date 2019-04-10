#Importing the liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import the dataset
dataset = pd.read_csv("Data.csv")
#matrix of features
x = dataset.iloc[:,:-1].values
#dependant var vector
y = dataset.iloc[:, -1].values

#handle missing data
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#handle categorical attributes
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#spliting the dataset into test_set and trainig_test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
scale_x = StandardScaler()
train_x = scale_x.fit_transform(train_x)
test_x = scale_x.transform(test_x)



