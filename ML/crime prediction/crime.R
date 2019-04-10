# Importing the dataset
dataset = read.csv("crime_data.csv")
dataset = dataset[2:3]

#split dataset into training set and test set
#library("caTools")
split = sample.split(dataset$Population, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting the Random Forest Model to the dataset
regressor = lm(formula = TotalCrimes ~ Population,
               data = training_set)

# Predicting a test set
y_pred = predict(regressor, newdata = test_set)

# Visualising the Model results(training set)
l#ibrary("ggplot2")
ggplot()+
  geom_point(aes(x = training_set$Population, y = training_set$TotalCrimes),
             colour = "red")+
  geom_line(aes(x = training_set$Population, y = predict(regressor, newdata = training_set)),
            colour = "blue")+
  ggtitle("Population vs Crimes Number (Training Set)")+
  xlab("Population")+
  ylab("Crimes Number")

# Visualising the Model results(training set)
ggplot()+
  geom_point(aes(x = test_set$Population, y = test_set$TotalCrimes),
             colour = "red")+
  geom_line(aes(x = training_set$Population, y = predict(regressor, newdata = training_set)),
            colour = "blue")+
  ggtitle("Population vs TotalCrimes (Test Set)")+
  xlab("Population")+
  ylab("Crime")



