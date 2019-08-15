#coding="utf-8"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = []
f = open('city.txt')
for v in f:
    X.append([float(v.split(',')[1]), float(v.split(',')[2])])

X = np.array(X)

n_clusters = 5

cls = KMeans(n_clusters).fit(X)

cls.labels_

markers = ['^', 'x', 'o', '*', '+']
for i in range(n_clusters):
  members = cls.labels_ == i
  plt.scatter(X[members, 0], X[members, 1], s=60, marker=markers[i], c='b', alpha=0.5)
plt.title(' ')
plt.show()
