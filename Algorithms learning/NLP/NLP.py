# Natural Language Processing

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the data
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
N = 1000
corpus = []
for i in range(0, N):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# create the bag of words
from sklearn.feature_extraction.text import CountVectorizer
countvectorizer = CountVectorizer(max_features = 1500)
X = countvectorizer.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

# split the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Apply the classifier model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(x_train, y_train)

# Predict the test set 
y_pred = classifier.predict(x_test)

# The confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)












