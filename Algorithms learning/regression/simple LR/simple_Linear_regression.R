# Simple Linear regression Model

#Importing the dataset
dataset = read.csv("Salary_Data.csv")

#spliting the dataset into test_set and trainig_set
#install.packages("caTools")
#library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fiting linear regression model to the training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#predicting the test set
y_pred = predict(regressor, newdata = test_set)

#visualising the training set
#install.packages("ggplot2")
#library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red')+
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue')+
  ggtitle("Salary vs Experience (Training set)")+
  xlab("Experience")+
  ylab("Salary")

#visualising the test set
ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red')+
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue')+
  ggtitle("Salary vs Experience (Test set)")+
  xlab("Experience")+
  ylab("Salary")
  
  
  

