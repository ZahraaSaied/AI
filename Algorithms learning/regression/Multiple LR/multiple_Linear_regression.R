#import dataset
dataset = read.csv("50_Startups.csv")

#handle categorical variable
dataset$State = factor(x = dataset$State,
                       levels = c("New York", "California", "Florida"),
                       labels = c(1, 2, 3))

#split the dataset into training set and test set
library("caTools")
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#build model
regressor = lm(formula = Profit ~ .,
               data = training_set)
#predict the test set
y_pred = predict(regressor, newdata = test_set)

#backward Elimnation
#first step (all in)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
#second step (eliminate variable with highest P value)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
#repeate second step
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)










