# K-Means Model

# Importing the dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5]

# Choosing best nomber of clusters by using Elbow method
set.seed(29)
wcss = vector()
for (i in 1:10) 
  {
    wcss[i] = sum(kmeans(X, i)$withinss) 
  } 
plot(1:10, wcss, type = 'b', main = paste("Elbow Method"), xlab = "No . of Clusters", ylab = "WCSS")

#Apply the K-Means model with 5 clusters
kmeans = kmeans(X, centers = 5, iter.max = 300, nstart = 10)
customers_clusters = kmeans$cluster

# Visualizing Clusters of Customers
library(cluster)
clusplot(X,
         customers_clusters,
         lines = 0, 
         labels = 0,
         shade = TRUE, 
         color = TRUE,
         span = TRUE,
         plotchar = FALSE,
         main = "Clusters of Customers",
         xlab = "Annual Income (K$)",
         ylab = "Score (1-10)")






