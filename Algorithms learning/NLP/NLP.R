# NLP
# Import the dataset
original_dataset = read.delim("Restaurant_Reviews.tsv", quote = "", stringsAsFactors = FALSE)

# Clean the data
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(original_dataset$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Create the bag of words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

# Build the dataset for the classification
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = original_dataset$Liked

#       Classification model
# Handle the dependent var as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# SPlit the dataset int training and test sets
library(caTools)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Build the classification model
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 50)

# Predict the test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Build the confusion matrix
cm = table(test_set[, 692], y_pred)






