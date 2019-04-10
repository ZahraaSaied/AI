# HC Model

# Import the dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[4:5] 

# Choosing best number of clusters by using dendrogram
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste("Dendrogram"),
     xlab = "Customers",
     ylab = "Euclidean distance")

# Fitting HC model with 5 clusters
hc = hclust(dist(X, method = 'euclidian'), method = 'ward.D')
customers_clusters = cutree(dendrogram, k = 5)

# Visualizing the clusters
library(cluster)
clusplot(X,
         customers_clusters,
         lines = 0,
         labels = 2,
         shade = TRUE,
         color = TRUE,
         span = TRUE,
         plotchar = FALSE,
         main = paste("customers clusters"),
         xlab = "Annual Salary (K$)",
         ylab = "Score (1-100)")












