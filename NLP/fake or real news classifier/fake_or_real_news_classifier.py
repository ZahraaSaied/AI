# Import the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
news_data = pd.read_csv('fake_or_real_news.csv')

# Explore the data
print(news_data.columns)
print(news_data.info())
print(news_data.head())

# Create a series to store the labels: y
y = news_data['label']
#type(y)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(news_data['text'], y, test_size = 0.25, random_state = 42)

# Initialize a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB(alpha = 0.1)

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score
score = accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

# Get the class labels
class_labels = nb_classifier.classes_

# Extract the features
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights
features_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], features_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], features_with_weights[-20:])
