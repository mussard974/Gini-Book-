"""
Gini KNN
--------
This module defines a rank-based Gini distance utility.

Overview:
    - Given a reference matrix X (n_train × d) and a query matrix X_new (n_test × d),
      it computes a Gini-style distance between each query row and all rows of X.
    - Ranks are computed columnwise and transformed with the generalized Gini
      parameter (gini_param ≥ 1), emphasizing tails as gini_param increases.

Key API:
    - GiniDistance(X, gini_param=2): store training/reference data.
    - compute_gini_ranks(X_new): conditional (decumulative) ranks for X and X_new.
    - gini_distance(x, Y, decum_rank_x, decum_ranks_Y): distance for one point.
    - compute_distances(X_new): full (n_test × n_train) distance matrix.

Notes:
    - Inputs must be NumPy arrays of shape (n_samples, n_features).
    - Distances are signed via rank-weighted differences; smaller values
      indicate closer rank-adjusted proximity.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy.linalg import inv, pinv
import scipy
import scipy.stats as ss

class GiniDistance:
    """
    Compute Gini distance between two matrix
    """
    def __init__(self, X, gini_param=2):
        self.X = X
        self.gini_param = gini_param

    def _rank(self, X):
        """Calculate the rank of X
        :input X : matrix over which to compute ranks
        """
        ranks = np.apply_along_axis(ss.rankdata, 0, X)
        return X.shape[0] - ranks + 1

    def compute_gini_ranks(self, X):
        """Compute conditional rank between X and self.X
        :input : X 
        """
        X_cat = np.concatenate((self.X, X), axis=0)
        ranks = (self._rank(X_cat) / X_cat.shape[0] * self.X.shape[0]) ** (self.gini_param - 1)
        return ranks[:self.X.shape[0]], ranks[self.X.shape[0]:]

    def gini_distance(self, x, Y, decum_rank_x, decum_ranks_Y):
        """Compute distance between a point and a matrix
        :input : x a vector 
        :input : Y a matrix
        :decum_rank_x : decumulative rank of vector x
        :decum_ranks_Y: decumulatice rank of matrix Y
        """
        distance = -np.sum((x - Y) * (decum_rank_x - decum_ranks_Y), axis=1)
        return distance

    def compute_distances(self, X):
        """Compute distance between the points of X and the points of self.X
        :input X a matrix
        """
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
