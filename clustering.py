"""

Unsupervised Learning

Clustering

"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


iris = load_iris()

data_iris = iris.data

"""

K means Clustering

"""

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters =3)

k_means.fit(data_iris)

labels = k_means.predict(data_iris)

print("labels",'\n', labels)

centers = k_means.cluster_centers_
print("centers",'\n', centers)

plt.scatter(data_iris[:,2], data_iris[:,3], c=labels)
plt.scatter(centers[:,2], centers[:,3], marker='o', color='red',s=120)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")

plt.show()

inertia = k_means.inertia_
print("inertia",inertia)

k_inertia = []

for i in range(1,10):
    k_means = KMeans(n_clusters =i,random_state= 44)
    k_means.fit(data_iris)
    k_inertia.append(k_means.inertia_)

print("k means inertia list",'\n', k_inertia)


plt.plot(range(1,10), k_inertia, color='green', marker='o')
plt.xlabel("Number of K")
plt.ylabel("Inertia")
plt.show()

"""

DBSCAN

"""

from sklearn.cluster import DBSCAN


DBS = DBSCAN(eps=0.7, min_samples=4) 

DBS.fit(data_iris)

labels = DBS.labels_

plt.scatter(data_iris[:,2], data_iris[:,3], c=labels)

plt.show()


"""

Hierarchical Clustering

"""

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

hr = linkage(data_iris, method='complete')

# dend = dendrogram(hr)

labels = fcluster(hr, 4,criterion='distance')
plt.scatter(data_iris[:,2], data_iris[:,3], c=labels)





plt.show()


