# HC Model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Choosing best number of clusters by using dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucludian distance")
plt.show()

# Fitting HC model with 5 clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
customers_clusters = hc.fit_predict(X)

# Visualizing customers clusters
plt.scatter(X[customers_clusters == 0, 0],
            X[customers_clusters == 0, 1],
            s = 100,
            color = 'magenta',
            label = 'Careful')
plt.scatter(X[customers_clusters == 1, 0], 
            X[customers_clusters == 1, 1],
            s = 100, 
            color = 'blue', 
            label = 'Normal')
plt.scatter(X[customers_clusters == 2, 0], 
            X[customers_clusters == 2, 1],
            s = 100, 
            color = 'red', 
            label = 'Target')
plt.scatter(X[customers_clusters == 3, 0], 
            X[customers_clusters == 3, 1],
            s = 100, 
            color = 'yellow', 
            label = 'careluss')
plt.scatter(X[customers_clusters == 4, 0], 
            X[customers_clusters == 4, 1],
            s = 100, 
            color = 'cyan', 
            label = 'Sensible')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()






