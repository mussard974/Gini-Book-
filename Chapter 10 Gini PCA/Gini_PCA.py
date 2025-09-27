"""
Gini PCA
--------
This module implements Gini-PCA, an extension of Principal Component Analysis 
that leverages Gini-based ranks and correlations instead of standard covariance. 
It provides tools for dimensionality reduction, outlier detection, correlation 
analysis, and visualization.

Features:
    - Compute Gini-based ranks, correlations, and distance matrices
    - Perform PCA using Gini distance metrics
    - Project and reconstruct data with Gini-based eigen decomposition
    - Evaluate eigenvalues and contributions of components
    - Detect outliers using Grubbs' test
    - Integrate categorical and mixed correlations (via Kendall’s tau)
    - Visualize results in 3D plots

Dependencies:
    - torch
    - numpy
    - pandas
    - scipy (scipy.stats, kendalltau)
    - scikit-learn (PCA, scaling utilities)
    - matplotlib (2D and 3D visualization)
    - iteration_utilities (deepflatten)
    - smirnov_grubbs (OUTLIERS / outliers package)

Class:
    GiniPca:
        Encapsulates methods for Gini-based PCA, correlation analysis, and 
        data visualization.
"""

import torch
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
try:
    from OUTLIERS import smirnov_grubbs as grubbs
except ImportError:
    from outliers import smirnov_grubbs as grubbs
from iteration_utilities import deepflatten

class GiniPca(object):
    
    def __init__(self, gini_param, n_components):
        self.gini_param = gini_param
        self.n_components = n_components

    def ranks(self, x):
        if type(x) == torch.Tensor:
            x = x.numpy()
        n, k = x.shape
        rank = np.zeros_like(x)
        for i in range(k):
            rank[:, i] = (n + 1 - ss.rankdata(x[:, i], method='average'))**(self.gini_param-1)
        rank = torch.tensor(rank).double()
        rank -= torch.mean(rank, axis=0)
        return rank

    def gmd(self, x):
        n, k = x.shape
        rank = self.ranks(x)
        xc = x - torch.mean(x, axis=0)
        GMD = -2/(n*(n - 1)) * self.gini_param * (xc.T @ rank)
        return GMD

    def scale_gini(self, x):
        G = self.gmd(x)
        z = (x - torch.mean(x, axis=0)) / torch.diag(G)
        return z

    def gini_correl(self, x):
        n, k = x.shape
        G = self.gmd(x)
        diag_G = torch.reshape(torch.diag(G), (k,1))
        return G / diag_G

    def fit(self, x):
        z = self.scale_gini(x)
        GMD = self.gmd(z)
        _, vecp = torch.linalg.eig(GMD.T + GMD)
        F = z @ torch.real(vecp)
        return F, vecp

    def preprocess_train_test(self, x_train, x_test):
        x_center = x_test - x_train.mean(dim=0, keepdim=True)
        self.mean_train_test = x_train.mean(dim=0, keepdim=True)
        return x_center.type(torch.float64)

    def fit_inverse(self, x_train, x_test):
        z = self.preprocess_train_test(x_train, x_test)
        GMD = self.gmd(z)
        _, vecp = torch.linalg.eig(GMD.T + GMD)        
        vecp = vecp.type(torch.float64)    
        F = z @ vecp
        F[:, self.n_components::] = 0
        F_inverse = F @ torch.inverse(vecp)
        F_recenter = F_inverse + self.mean_train_test
        return F_recenter

    def eigen_val(self, x):
        z = self.scale_gini(x)
        GMD = self.gmd(z)
        valp, _ = torch.linalg.eig(GMD.T + GMD)
        A = torch.real(valp) / torch.real(valp).sum() * 100
        return A

    def absolute_contrib(self, x):
        n, k = x.shape
        z = self.scale_gini(x)
        GMD = self.gmd(z)
        valp, vecp = torch.linalg.eig(GMD.T + GMD)
        F = z @ torch.real(vecp)
        rank_z = self.ranks(z)
        return ((-2/(n*(n-1)))* self.gini_param * (F*(rank_z @ torch.real(vecp)))) / (torch.real(valp)/2) 

    def relative_contrib(self, x):
        F = self.project(x)
        return torch.abs(F) / torch.sum(abs(F), axis=0)

    def gini_correl_axis(self, x):
        n, k = x.shape
        z = self.scale_gini(x)
        F = self.fit(x)[0]
        rank_x = self.ranks(x)
        diag = torch.reshape(torch.diag(z.T @ rank_x), (k, 1))
        return (F.T @ rank_x) / diag

    def u_stat(self, x):
        n, k = x.shape
        F = self.fit(x)[0]
        GC = self.gini_correl_axis(x)
        z = self.scale_gini(x)
        axis_1 = torch.zeros_like(F)
        axis_2 = torch.zeros_like(F)
        for i in range(n):
            F_i = torch.cat((F[:i,:], F[i+1:,:]), axis=0)
            z_i = torch.cat((z[:i,:], z[i+1:,:]), axis=0) 
            rank_z_i = self.ranks(z_i)
            diag = torch.reshape(torch.diag(z_i.T @ rank_z_i), (k, 1)) 
            gini_cor = (F_i.T @ rank_z_i) / diag
            axis_1[i, :] = gini_cor[0, :]
            axis_2[i, :] = gini_cor[1, :]
        std_jkf = torch.zeros((2, k))
        std_jkf[0, :] = torch.sqrt(torch.var(axis_1, axis=0, correction=1) * ((n - 1)**2 / n))
        std_jkf[1, :] = torch.sqrt(torch.var(axis_2, axis=0, correction=1) * ((n - 1)**2 / n))
        ratio = GC[:2, :] / std_jkf
        return ratio

    def optimal_gini_param(self, x):
        if type(x) == torch.Tensor:
            x_copy = x.numpy()
        n, k = x.shape
        a = []
        for i in range(k):
            a.append(grubbs.max_test_indices(x_copy[:, i], alpha=0.05))
        x_outlier = np.delete(x_copy, list(deepflatten(a)), axis=0) 
        x_outlier = torch.tensor(x_outlier).double()
        eigen_val = []
        for self.gini_param in np.arange(1.1, 6, 0.1):
            z_outlier = self.scale_gini(x_outlier)
            GMD_outlier = self.gmd(z_outlier)
            valp_outlier, _ = torch.linalg.eig(GMD_outlier.T + GMD_outlier)
            z = self.scale_gini(x)
            GMD = self.gmd(z)
            valp, _ = torch.linalg.eig(GMD.T + GMD)
            eigen_val.append(torch.abs(torch.real(valp[:2].sum())/torch.real(valp).sum() - torch.real(valp_outlier[:2].sum())/torch.real(valp_outlier).sum()))
        if (torch.argmin(torch.tensor(eigen_val))+1)/10 == 1:
            gini_param = (torch.argmin(torch.tensor(eigen_val))+1)/10 + 0.1
        else:
            gini_param = (torch.argmin(torch.tensor(eigen_val))+1)/10
        return gini_param

    def plot3D(self, x):
        x_copy = torch.DoubleTensor(x.values)
        n, k = x.shape
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
        F = self.fit(x_copy)[0]
        for line in range(n):
            ax.scatter(F[line, 0], F[line, 1], F[line, 2], color='b', edgecolor='k', s=40)
            ax.text(F[line, 0], F[line, 1], F[line, 2], x.index[line], size=10, color='k')
        ax.set_xlabel("1st component")
        ax.set_xticklabels([])
        ax.set_ylabel("2nd component")
        ax.set_yticklabels([])
        ax.set_zlabel("3rd component")
        ax.set_zticklabels([])
        return plt.show()    
    
    def outliers(self, x):
        if type(x) == torch.Tensor:
            x = x.numpy()
        n, k = x.shape
        outliers_variables = []
        for i in range(1, k):
            outliers_variables.append(grubbs.max_test_indices(x[:, i], alpha=0.1))
        if len(outliers_variables) > 0:
            self.number_outliers = len(outliers_variables)
            return self.number_outliers

    # Ajout des nouvelles fonctions pour calculer les corrélations catégorielles et mixtes
    def calculate_categorical_correlations_kendall(self, df, categorical_columns):
        corr_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                                   index=categorical_columns, columns=categorical_columns)
        for col1 in categorical_columns:
            for col2 in categorical_columns:
                if col1 != col2:
                    corr_matrix.loc[col1, col2], _ = kendalltau(df[col1], df[col2])
        return corr_matrix

    def calculate_mixed_correlations_kendall(self, df, quantitative_columns, categorical_columns):
        corr_matrix = pd.DataFrame(np.zeros((len(quantitative_columns), len(categorical_columns))), 
                                   index=quantitative_columns, columns=categorical_columns)
        for quant_col in quantitative_columns:
            for cat_col in categorical_columns:
                corr_matrix.loc[quant_col, cat_col], _ = kendalltau(df[quant_col], df[cat_col])
        return corr_matrix

    # Intégration de toutes les corrélations
    def integrate_all_correlations(self, df, quantitative_columns, categorical_columns):
        x = torch.tensor(df[quantitative_columns].values, dtype=torch.float64)
        gmd_matrix = self.gmd(x)
        cat_corr_matrix = self.calculate_categorical_correlations_kendall(df, categorical_columns)
        mixed_corr_matrix = self.calculate_mixed_correlations_kendall(df, quantitative_columns, categorical_columns)
        
        combined_matrix = pd.concat([
            pd.concat([pd.DataFrame(gmd_matrix.numpy(), index=quantitative_columns, columns=quantitative_columns), mixed_corr_matrix], axis=1),
            pd.concat([mixed_corr_matrix.T, cat_corr_matrix], axis=1)
        ], axis=0)
        
        return combined_matrix

    # Calcul des valeurs propres et vecteurs propres
    def calculate_eigenvalues_and_vectors(self, correlation_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
        return eigenvalues, eigenvectors

