# Apriori Model

# Import the dataset
library(arules)
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 50)

# Build our model
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2, minlen = 2))

# Visualise the top rules
inspect(sort(rules, by = 'lift')[1:10])
