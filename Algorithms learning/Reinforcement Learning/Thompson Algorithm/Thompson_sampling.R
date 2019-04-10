# Thompson Sampling Algorithm

# Import the dataset
dataset = read.csv("Ads_CTR_Optimisation.csv")

# Build the model
N = 10000
d = 10
number_of_one_rewards = integer(d)
number_of_zero_rewards = integer(d)
selected_ads = integer()
total_rewards = 0

for (n in 1:N) {
  max_random = 0
  ad = 0
  for (i in 1:d) {
    random_beta = rbeta(n = 1,
                        shape1 = number_of_one_rewards[i] + 1,
                        shape2 = number_of_zero_rewards[i] + 1)
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  
  selected_ads = append(selected_ads, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    number_of_one_rewards[ad] = number_of_one_rewards[ad] + 1
  } else {
    number_of_zero_rewards[ad] = number_of_zero_rewards[ad] + 1
  }
    
  total_rewards = total_rewards + reward
}

# visualise the results
hist(selected_ads,
     col = 'purple',
     main = "Histogram of the Selected Ads",
     xlab = "Ads",
     ylab = "Number of each Ad selections")


