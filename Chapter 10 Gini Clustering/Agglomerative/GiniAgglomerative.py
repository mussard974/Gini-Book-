"""
Gini Distance for Agglomerative Clustering
--------
This module provides an implementation of Gini-based distances and related 
clustering utilities. It is designed to extend traditional distance metrics by 
leveraging Gini ranks, allowing for more robust handling of non-Gaussian data 
and outliers.

Features:
    - Compute Gini-based ranks and distances between matrices
    - Construct full pairwise distance matrices
    - Detect outliers using Grubbs’ statistical test
    - Map cluster labels to ground-truth labels via confusion matrices
    - Visualize hierarchical clustering with dendrograms

Dependencies:
    - numpy
    - pandas
    - scipy (stats, hierarchical clustering, Grubbs’ test via outliers/OUTLIERS package)
    - scikit-learn (AgglomerativeClustering, metrics)
    - matplotlib (dendrogram plotting)

Class:
    GiniDistance:
        Encapsulates methods for computing Gini ranks, distances, 
        outlier detection, label mapping, and dendrogram visualization.
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.stats as ss
import pandas as pd
try:
    from OUTLIERS import smirnov_grubbs as grubbs
except ImportError:
    from outliers import smirnov_grubbs as grubbs
    
class GiniDistance:
    """
    Compute Gini prametric (distance) between two matrices
    """
    def __init__(self, X, gini_param=2):
        self.X = X
        self.gini_param = gini_param

    def _rank(self, X):
        "Calculate the rank of X"
        ranks = np.apply_along_axis(ss.rankdata, 0, X)
        return X.shape[0] - ranks + 1

    def compute_gini_ranks(self, X):
        "Compute conditional rank between X and self.X"
        X_cat = np.concatenate((self.X, X), axis=0)
        ranks = (self._rank(X_cat) / X_cat.shape[0] * self.X.shape[0]) ** (self.gini_param - 1)
        return ranks[:self.X.shape[0]], ranks[self.X.shape[0]:]

    def gini_distance(self, x, Y, decum_rank_x, decum_ranks_Y):
        "Compute distance between a point and a matrix"
        distance = -np.sum((x - Y) * (decum_rank_x - decum_ranks_Y), axis=1)
        return distance

    def compute_distances(self, X):
        "Compute distance between the points of X and the points of self.X"
        ranks_train, ranks_test = self.compute_gini_ranks(X)  
        n_test = X.shape[0]
        n_train = self.X.shape[0]
        distances = np.zeros((n_test, n_train))
        
        for i, x in enumerate(X):
            distances[i, :] = self.gini_distance(
                x, self.X,
                ranks_test[i],
                ranks_train
            )
        return distances
    
    def outliers(self, x):
        outliers_numbers = []
        for i in range (x.shape[1]):
            outliers_numbers.append(len(grubbs.max_test_indices(x[:,i], alpha = 0.1)))
        outliers_numbers = np.array(outliers_numbers)
        return outliers_numbers, outliers_numbers.sum()

    def map_labels(self, cluster_labels, true_labels):
        confusion = confusion_matrix(true_labels, cluster_labels)
        label_mapping = {}
        for cluster_id in range(confusion.shape[1]):
            true_label = np.argmax(confusion[:, cluster_id])
            label_mapping[cluster_id] = true_label
        # Map
        mapped_labels = np.array([label_mapping[label] for label in cluster_labels])
        return mapped_labels
    
    def plot_dendrogram(self, model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)
        
