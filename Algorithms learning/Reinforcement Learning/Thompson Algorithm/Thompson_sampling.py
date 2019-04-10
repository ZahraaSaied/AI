# Thompson Sampling Algorithm

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Build the Algorithm
import random
N = 10000
d = 10
numbers_of_zero_rewards = [0] * d
numbers_of_one_rewards = [0] * d
selected_ads = []
total_reward = 0

for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_one_rewards[i] + 1, numbers_of_zero_rewards[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    selected_ads.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_one_rewards[ad] = numbers_of_one_rewards[ad] + 1
    else:
        numbers_of_zero_rewards[ad] = numbers_of_zero_rewards[ad] + 1
    total_reward += reward
        
# Visualise the results
plt.hist(selected_ads)
plt.title("The Histogram of Selected Ads")
plt.xlabel("Ads")
plt.ylabel("Number Of Selections")
