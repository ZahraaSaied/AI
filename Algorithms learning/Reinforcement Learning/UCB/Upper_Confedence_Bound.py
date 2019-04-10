# Upper Confedence Bound Algorithm

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Build the Algorithm
import math
N = 10000
d = 10
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
selected_ads = []
total_rewards = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0) :
            average_reward = sum_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    selected_ads.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_rewards += reward

# Visualise the histogram of selected ads
plt.hist(selected_ads)
plt.title("The Histogram of Selected Ads")
plt.xlabel("Ads")
plt.ylabel("Number Of Selections")
