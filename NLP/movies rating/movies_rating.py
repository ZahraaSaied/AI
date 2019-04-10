# Movies Rating analysis


# Import the libraries
import numpy as np
import pandas as pd
import matplotlib as plt

# Import the dataset
dataset = pd.read_csv("imdb_labelled.txt", delimiter = "\t", quoting = 3, header = None)

# Cleaning the data
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

N = 1000
corpus = []

for i in range(0, N):
    review = dataset[0][i]
    review = re.sub('[^A-Za-z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create the bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(max_features = 2000)
X = count_vectorizer.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

# Split the dataset to train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Apply the classifier model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, 
                                    criterion = 'entropy',
                                    random_state = 20)
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluate model using k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, 
                             X = x_train,
                             y = y_train,
                             scoring = 'accuracy',
                             cv = 10,
                             n_jobs = -1)
average_accuracy = accuracies.mean()
variance = accuracies.std()








    
    