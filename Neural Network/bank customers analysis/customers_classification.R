# Artifitial Neural Network

# Import the dataset
dataset = read.csv("Churn_Modelling.csv")
dataset = dataset[4:14]

# Handle categorial variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                           levels = c('France', 'Spain', 'Germany'),
                           labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                        levels = c('Male', 'Female'),
                        labels = c(1, 2)))

# Split the dataset into training and test sets
library(caTools)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

###  Build ANN  ###
library(h2o)
#initialize h2o instance
h2o.init(nthreads = -1)

#build and fit the ANN
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6, 6),
                              epochs = 100,
                              train_samples_per_iteration = -2)


# Predict the test set results
y_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# The Confusion Matrix
cm = table(test_set[, 11] ,y_pred)










