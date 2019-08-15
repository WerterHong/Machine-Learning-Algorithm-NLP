#  encoding="utf-8"

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DBScan (object):
    def __init__(self, p, l_stauts):

        self.point = p
        self.labels_stats = l_stauts
        self.db = DBSCAN(eps=0.2, min_samples=10).fit(self.point)

    def draw(self):

        coreSamplesMask = np.zeros_like(self.db.labels_, dtype=bool)
        coreSamplesMask[self.db.core_sample_indices_] = True
        labels = self.db.labels_
        nclusters = jiangzao(labels)

        print('Estimated number of clusters: %d' % nclusters)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.labels_stats, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(self.labels_stats, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.labels_stats, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(self.labels_stats, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(self.labels_stats, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.point, labels))

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'

            classMemberMask = (labels == k)

            xy = self.point[classMemberMask & coreSamplesMask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

            xy = self.point[classMemberMask & ~coreSamplesMask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=3)

        plt.title('Estimated number of clusters: %d' % nclusters)
        plt.show()


def jiangzao (labels):

    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return clusters

def standar_scaler(points):

    p = StandardScaler().fit_transform(points)
    return p

if __name__ == "__main__":

     centers = [[1, 1], [-1, -1], [-1, 1], [1, -1]]
     point, labelsTrue = make_blobs(n_samples=2000, centers=centers, cluster_std=0.4,
                                    random_state=0)
     point = standar_scaler(point)
     db = DBScan(point, labelsTrue)
     db.draw()
