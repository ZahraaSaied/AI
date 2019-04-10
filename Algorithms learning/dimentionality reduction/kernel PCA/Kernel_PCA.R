# Kernel PCA algorithm

# Import the dataset 
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[3:5]

# Split the dataset into training and test sets
library(caTools)
set.seed(12)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Applying Kernel PCA for feature extraction
library(kernlab)
kpca = kpca(~.,
            data = training_set[-3],
            kernel = "rbfdot",
            features = 2)

training_set_pca =as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased

test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

# Apply Logistic Regression for the classification task
classifier = glm(formula = Purchased ~ .,
                 family = "binomial",
                 data = training_set_pca)

# predict the test set results
y_pred = predict(classifier,type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(y_pred > 0.5, 1, 0)
# The confusion Matrix
cm = table(test_set_pca[, 3], y_pred)

# Visualising the Training set results
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Kernel PCA Feature 1', ylab = 'Kernel PCA Feature 2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Kernel PCA Feature 1', ylab = 'Kernel PCA Feature 2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))





