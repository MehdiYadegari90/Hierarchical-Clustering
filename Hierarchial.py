# Importing necessary libraries
import pandas as pd
import scipy.spatial.distance
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import pylab

# Load the dataset
df = pd.read_csv('Clustering2.csv')

# Drop the 'Gender' column as it's not needed for clustering
df = df.drop("Gender", axis=1)

# Convert all columns to numeric and drop rows with NaN values
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()
df = df.reset_index(drop=True)

# Normalize the data using Min-Max scaling
x = df.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)

# Create a distance matrix using Euclidean distance
leng = feature_mtx.shape[0]
D = np.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

# Perform hierarchical clustering
Z = hierarchy.linkage(D, "complete")

# Define a function for labeling dendrogram leaves
def llf(id):
    return "[%s %s %s]" % (df["Age"][id], df["Annual Income (k$)"][id], df["Spending Score (1-100)"][id])

# Plot the dendrogram
fig = pylab.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, leaf_font_size=7)
plt.show()

# Perform Agglomerative Clustering with a specified number of clusters
agglom = AgglomerativeClustering(n_clusters=4, linkage="complete")
agglom.fit(D)

# Assign cluster labels to the original DataFrame
df["cluster"] = agglom.labels_

# Scatter plot of the clusters based on Age and Income
plt.scatter(x[:, 0], x[:, 1], c=agglom.labels_.astype(float), alpha=1)
plt.xlabel("Age", fontsize=18)
plt.ylabel("Income", fontsize=16)

# Add cluster labels to the scatter plot
for i in range(len(x)):
    plt.text(x[i, 0] + 0.2, x[i, 1] + 0.2, str(agglom.labels_[i]), rotation=25)

plt.show()

# 3D scatter plot of the clusters
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
ax.set_xlabel("Edu")
ax.set_ylabel("Age")
ax.set_zlabel("Income")
ax.scatter(x[:, 1], x[:, 0], x[:, 2], c=agglom.labels_)

# Show the 3D plot
plt.show()
