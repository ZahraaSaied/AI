# Classification with Artifitial Neural Network

# Import the dataset
import pandas as pd
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
country_encoder = LabelEncoder()
X[:, 1] = country_encoder.fit_transform(X[:, 1])
gender_encoder = LabelEncoder()
X[:, 2] = gender_encoder.fit_transform(X[:, 2])
hot_encoder = OneHotEncoder(categorical_features = [1])
X = hot_encoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split the dataset int training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
x_train = x_sc.fit_transform(x_train)
x_test = x_sc.transform(x_test)

## Build Neural Network ##
# Import the libraries
from keras.models import Sequential
from keras.layers import Dense

# Initialize our network
classifier = Sequential()

# Build both the input and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# The output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Configure the learning process
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit our ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

# Predict the test set result
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

