# Apriori Model

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Build the Apriori Model
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_lenght = 2)

# Visualise the rules
results = list(rules)

""" print("number of rules", len(results))
for i in range(0, 10):
    result = results[i]
    supp = int(result.support*10000)/100
    conf = int(result.ordered_statistics[0].confidence*100)
    item1 = ''.join([x+' ' for x in result.ordered_statistics[0].items_base])
    item2 = ''.join([x+' ' for x in result.ordered_statistics[0].items_add])
    print("if "+str(item1)+ "is purchased, then the "+str(item2)+"has a chance of "+str(conf)+"% to be purchased [support = "+str(supp)+"%]")
    """
