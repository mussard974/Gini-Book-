import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
from numpy.linalg import inv
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import torch

class GiniKmeans(object):
    
    '''
    Gini Kmeans
    ---------
    Variables
    ---------
        n_clusters = K = number of clusters
        max_iter = maximum number of iterations
    ---------    
    Functions
    ---------
        x --> matrix with data
        euclidean(point,x): return Euclidean distance
        fit(x): fit Kmeans with the Euclidean distance
        evaluate(x): return centroids and labels
        gini_distance(point,x): Gini distance function
        fit_gini(x): fit Kmeans in the Gini pseudo-metric space
        evaluate_gini(x): return centroids and labels
    '''
    
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter    
    
    def euclidean(self, point, X):
        return np.sqrt(np.sum((point - X)**2, axis=1))

    # Part of the code is inspired by a script of Luke Turner, see [https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670]
    def fit(self, X):        
        _,n_dim =  X.shape
        min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
        self.centroids = [random.uniform(min_, max_) for _ in range(self.n_clusters)]   
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X:
                distance_l2 = self.euclidean(x, self.centroids)
                centroid_idx = np.argmin(distance_l2)
                sorted_points[centroid_idx].append(x)            
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        return self.centroids

    def evaluate(self, X):
        centroid_idx = []
        for x in X:
            distance_l2 = self.euclidean(x, self.centroids)
            centroid_idx.append(np.argmin(distance_l2))        
        return self.centroids, centroid_idx
    
    def gini_distance(self, point, center):
        rank_point = ss.rankdata(point, method='average')
        list_rank = []
        for i in range(self.n_clusters):
            list_rank.append(ss.rankdata(center[i], method='average'))
        return np.sum((point - center)*(rank_point - list_rank), axis=1)

    def fit_gini(self, X):        
        _,n_dim =  X.shape
        min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
        self.centroids_gini = [random.uniform(min_, max_) for _ in range(self.n_clusters)]   
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids_gini, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)] # list to gather each point with its nearest centroid
            for x in X:
                distance_gini = self.gini_distance(x, self.centroids_gini)
                centroid_idx = np.argmin(distance_gini)
                sorted_points[centroid_idx].append(x)             # each point put in its close group  
            prev_centroids = self.centroids_gini
            self.centroids_gini = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids_gini):
                if np.isnan(centroid).any(): 
                    self.centroids_gini[i] = prev_centroids[i]
            iteration += 1
        return self.centroids_gini
    
    def evaluate_gini(self, X):
        centroid_idx = []
        for x in X:
            dist_gini = self.gini_distance(x, self.centroids_gini)
            centroid_idx.append(np.argmin(dist_gini))                  
        return self.centroids_gini, centroid_idx  
