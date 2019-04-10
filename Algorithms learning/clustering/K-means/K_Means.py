# K-Means Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Choosing best nomber of clusters by using Elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("No. Of Clusters")
plt.ylabel("WCSS")
plt.show()

# Applying K-means to the dataset with 5 clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_clusters = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_clusters == 0, 0], 
            X[y_clusters == 0, 1], 
            s = 100, 
            color = 'gray', 
            label = 'Careful')
plt.scatter(X[y_clusters == 1, 0], 
            X[y_clusters == 1, 1], 
            s = 100, 
            color = 'blue', 
            label = 'Normal')
plt.scatter(X[y_clusters == 2, 0], 
            X[y_clusters == 2, 1], 
            s = 100, 
            color = 'red', 
            label = 'Target')
plt.scatter(X[y_clusters == 3, 0], 
            X[y_clusters == 3, 1], 
            s = 100, 
            color = 'magenta', 
            label = 'Careless')
plt.scatter(X[y_clusters == 4, 0], 
            X[y_clusters == 4, 1], 
            s = 100, 
            color = 'green', 
            label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s = 200, color = 'yellow', 
            label = 'Centroides')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()