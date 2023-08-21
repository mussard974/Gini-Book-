import torch
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from outliers import smirnov_grubbs as grubbs
from iteration_utilities import deepflatten

class GiniPca(object):
    
    '''
    Gini PCA
    ---------
    Variables
    ---------
        gini_param : Gini parameter
    ---------    
    Functions
    ---------
        x --> matrix with data
        ranks(x): return the rank vectors of each columns
        gmd(x): return the Gini Mean Difference matrix between variables
        scale_gini(x): return the standardization with the GMD instead of the variance
        gini_correl(x): return the matrix of the Gini correlation between variables
        project(x): return the projection of the variables into the Gini subspace 
        eigen_val(x): return eigenvalues
        absolute_contrib(x): return the absolute contributions of the points
        relative_contrib(x): return the Gini distance of the points to the principal components
        gini_correl_axis(x): return the correlation of the variables with the principal components
        u_stat(x): return U statitics to test for the significance of the variables on the two first principal components
        optimal_gini_param(x): return the optimal Gini parameter
    '''
    
    def __init__(self, gini_param):
        
        self.gini_param = gini_param
    
    def ranks(self, x):
        #torch.tensor(x).double()
        if type(x) == torch.Tensor:
            x = x.numpy()
        n, k = x.shape
        rank = np.zeros_like(x)
        for i in range(k):
        #r = (n-torch.argsort(x, dim=0, descending=False, stable=True))**(self.gini_param-1) # Returns long tensor
            rank[:,i] = (n + 1 - ss.rankdata(x[:,i], method='average'))**(self.gini_param-1)
        rank = torch.tensor(rank).double() # convert to float64
        rank -= torch.mean(rank, axis=0)
        return rank
    
    def gmd(self, x):
        if type(x) == np.ndarray:
            x = torch.tensor(x).double()
        n, k = x.shape
        rank = self.ranks(x)
        xc = x - torch.mean(x, axis=0)
        GMD = -2/(n*(n - 1)) * self.gini_param * (xc.T @ rank)
        return GMD
    
    def scale_gini(self, x):
        if type(x) == np.ndarray:
            x = torch.tensor(x).double()
        G = self.gmd(x)
        z = (x - torch.mean(x, axis=0)) / torch.diag(G)
        return z

    def gini_correl(self, x):
        n, k = x.shape
        G = self.gmd(x)
        diag_G = torch.reshape(torch.diag(G), (k,1)) # transpose diag
        return G / diag_G
    
    def project(self, x):
        z = self.scale_gini(x)
        GMD = self.gmd(z)
        _, vecp = torch.linalg.eig(GMD.T + GMD)
        F = z @ torch.real(vecp)
        return F
    
    def eigen_val(self, x):
        z = self.scale_gini(x)
        GMD = self.gmd(z)
        valp, _ = torch.linalg.eig(GMD.T + GMD)
        return torch.real(valp) / torch.real(valp).sum() * 100
    
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
        return torch.abs(F) / torch.sum(abs(F), axis = 0)
    
    def gini_correl_axis(self, x):
        n, k = x.shape
        z = self.scale_gini(x)
        F = self.project(x)
        rank_x = self.ranks(x)
        diag = torch.reshape(torch.diag(z.T @ rank_x), (k,1))
        return (F.T @ rank_x) / diag
    
    def u_stat(self, x):
        n, k = x.shape
        F = self.project(x)
        GC = self.gini_correl_axis(x)
        z = self.scale_gini(x)
        axis_1 = torch.zeros_like(F)
        axis_2 = torch.zeros_like(F)
        for i in range(n):
            F_i = torch.cat((F[:i,:], F[i+1:,:]), axis = 0)
            z_i = torch.cat((z[:i,:], z[i+1:,:]), axis = 0) 
            rank_z_i = self.ranks(z_i)
            diag = torch.reshape(torch.diag(z_i.T @ rank_z_i), (k,1)) 
            gini_cor = (F_i.T @ rank_z_i) / diag
            axis_1[i,:] = gini_cor[0,:]
            axis_2[i,:] = gini_cor[1,:]
        std_jkf = torch.zeros((2, k))
        std_jkf[0,:] = torch.sqrt(torch.var(axis_1, axis =0, correction=1) * ((n - 1)**2 / n))
        std_jkf[1,:] = torch.sqrt(torch.var(axis_2, axis =0, correction=1) * ((n - 1)**2 / n))
        ratio = GC[:2, :] / std_jkf
        return ratio
    
    def optimal_gini_param(self,x):
        if type(x) == torch.Tensor:
            x_copy = x.numpy()
        n, k = x.shape
        a=[]
        for i in range (k):
            a.append(grubbs.max_test_indices(x_copy[:,i], alpha=0.05))
        x_outlier = np.delete(x_copy, list(deepflatten(a)), axis=0) 
        x_outlier = torch.tensor(x_outlier).double() # convert to torch float 64
        eigen_val = []
        for self.gini_param in np.arange(1.1, 6, 0.1):
            z_outlier = self.scale_gini(x_outlier)
            GMD_outlier = self.gmd(z_outlier)
            valp_outlier,_ = torch.linalg.eig(GMD_outlier.T + GMD_outlier)
            z = self.scale_gini(x)
            GMD = self.gmd(z)
            valp,_ = torch.linalg.eig(GMD.T + GMD)
            eigen_val.append(torch.abs(torch.real(valp[:2].sum())/torch.real(valp).sum() - valp_outlier[:2].sum()/torch.real(valp_outlier).sum()))
        if (torch.argmin(torch.tensor(eigen_val))+1)/10 == 1:
            gini_param = (torch.argmin(torch.tensor(eigen_val))+1)/10 + 0.1
        else:
            gini_param = (torch.argmin(torch.tensor(eigen_val))+1)/10
        return gini_param
    
