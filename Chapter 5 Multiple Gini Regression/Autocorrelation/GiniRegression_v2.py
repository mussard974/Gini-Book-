import torch
from datetime import datetime
from scipy.stats import norm
from statsmodels.iolib.table import SimpleTable
from statsmodels.compat.python import lrange
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
try:
    from OUTLIERS import smirnov_grubbs as grubbs
except ImportError:
    from outliers import smirnov_grubbs as grubbs
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from numpy.linalg import pinv
from scipy import linalg
from sklearn.metrics import confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.optimize import minimize
import warnings
import scipy.linalg as la
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from scipy import stats


class GiniRegression(object):
    
    '''
    Gini Regression
    Variables :
    
    ----------
    Functions:
    ----------
    fit :
        - add_constant (bool) : True whenever the user has already adds the first column with constant. Default is False 

        - iter (float) : fix the sample size for the std.dev. Jackknife estimator (non parametric method only)
         
        - parametric_estimator (bool) : default False in this case the non parametric estimator is used 
            If parametric_estimator = True : solver that minimizes the co-Gini of the errors
        
        - cov : variance-cov matrix to parametric estimation only: 
            => Default is the jackknife variance of the non parametric beta Gini estimators 
            => 'asympt' for asymptotic estimation like in OLS : sigma_e^2 (X'X)^{-1}
            => 'instrument' for asymptotic estimation like in OLS with rank_x = Z as instrument : sigma_e^2 (Z'X)^{-1} @ (X'X) @ (X'Z)^{-1}
            => 'bandwidth iid' see statsmodels fo quantile regressions : https://www.statsmodels.org/dev/_modules/statsmodels/regression/quantile_regression.html
        
        - gini_param for generalized Gini regression
    '''
    
    def __init__(self, gini_param = False, parametric_estimator = False, cov = False, iter = False):
        self.parametric_estimator = parametric_estimator
        self.cov = cov
        self.iter = iter
        self.gini_param = gini_param 
    
    def format_x(self, x):
        if isinstance(x, pd.DataFrame):
            return x.values  
        elif isinstance(x, pd.Series):
            return x.to_numpy()  
        elif isinstance(x, torch.Tensor):
            return x.numpy() 
        else:
            return x 

    
    def format_y(self, y):
        if isinstance(y, pd.DataFrame):
            return y.to_numpy().flatten()
        elif isinstance(y, pd.Series):
            return y.to_numpy().flatten()
        elif isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy().flatten()
        else:
            return np.array(y).flatten()
                
    def x_names(self, x):
        if isinstance(x, pd.DataFrame):
            if x.columns.names:
                return list(x.columns)
            else:
                return [f"x{i+1}" for i in range(x.shape[1])]
        elif isinstance(x, np.ndarray):
            if x.ndim == 1:  # Si c'est un vecteur 1D
                return [f"x1"]
            elif x.ndim == 2:  # Si c'est un tableau 2D
                return [f"x{i+1}" for i in range(x.shape[1])]
            else:
                return ["x1"]
        else:
            try:
                return [f"x{i+1}" for i in range(x.shape[1])]
            except (IndexError, AttributeError):
                return ["x1"]  # Valeur par défaut si x n'a pas de forme exploitable

    def y_names(self, y):
        if isinstance(y, pd.DataFrame):
            if y.columns.names:
                return y.columns
        else:
            return "y"

    def ranks(self, x):
        x = self.format_x(x)
        n, k = x.shape
        rank = np.zeros_like(x)
        if not self.gini_param:
            for i in range(k):
                rank[:,i] = ss.rankdata(x[:,i], method='average')
        else:            
            for i in range(k):
                rank[:,i] = n+1 - ss.rankdata(x[:,i], method='average')
                rank[:,i] = rank[:,i]**(self.gini_param-1) 
        return rank

    def gini_estimator(self, y, x): # x must contain the column for the constant
        x = self.format_x(x)
        y = self.format_y(y)
        rank_x = self.ranks(x)
        beta_gini = inv(rank_x.T@x)@(rank_x.T@y)
        return beta_gini

    def gini_estimator_parametric(self, y, x):
        x = self.format_x(x)
        y = self.format_y(y)
        y = y.reshape(-1, 1)
        if self.x.ndim >=2:
            n,_ = x.shape
        else:
            n = x.shape[0]
        X = np.concatenate((y,np.ones((n,1)), x), axis = 1) 
        if not self.gini_param:
            def gini_param_(coef, X_data):
                residuals = X_data[:, 0] - (coef * X_data[:, 1:]).sum(axis=1)
                rank_residuals = ss.rankdata(residuals, method='average')
                return 1/n*np.cov(residuals.T,rank_residuals.T)[0][1] # function to minimize
        else:
            def gini_param_(coef, X_data):
                residuals = X_data[:, 0] - (coef * X_data[:, 1:]).sum(axis=1)
                n,_ = X_data.shape
                rank_residuals = (n+1-ss.rankdata(residuals, method='average'))**(self.gini_param-1)
                return 1/n*np.cov(residuals.T,rank_residuals.T)[0][1] # function to minimize
        # Output of gini_param
        initial = [0] * (x.shape[1]+1)
        result = minimize(gini_param_, initial, args=(X,))
        beta_gini_parametric = result.x
        return beta_gini_parametric

    def gini_R2(self, y, x): # x must contain the constant 
        x = self.format_x(x)
        y = self.format_y(y)
        beta_coeff = self.gini_estimator(y, x)
        residuals = y - x@beta_coeff
        rank_y = ss.rankdata(y, method='average')
        rank_residuals = ss.rankdata(residuals, method='average')
        return 1 - (np.cov(residuals.T,rank_residuals.T)[0][1]**2) / (np.cov(y.T,rank_y.T)[0][1]**2)

    def variance_inflation_factor(self, x, x_idx):
        x = self.format_x(x)
        n,_ = x.shape
        k_vars = x.shape[1]
        x_i = x[:, x_idx]
        mask = np.arange(k_vars) != x_idx
        x_noti = x[:, mask]
        x_noti = np.c_[np.ones(n), x_noti]
        gini_r_squared_i = self.gini_R2(x_i, x_noti) 
        vif = 1. / (1. - gini_r_squared_i)
        return np.round(vif,4)

    def hall_sheather(self, n, q, alpha=.05): # bandwidth hall_sheather as in statsmodels
        z = norm.ppf(q)
        num = 1.5 * norm.pdf(z)**2.
        den = 2. * z**2. + 1.
        h = n**(-1. / 3) * norm.ppf(1. - alpha / 2.)**(2./3) * (num / den)**(1./3)
        return h

    def my_warning(self):
        warnings.warn("If no cov method is invoked then the non-parametric cov is used", UserWarning)      

    def fit(self, y, x, add_constant = False, parametric_estimator = False, cov = False, iter = 1000): # x with constant       
        # add_constant = True means that the user has put a column of constants 
        self.parametric_estimator = parametric_estimator
        self.cov = cov
        self.iter = iter
        self.add_constant = add_constant
        
        if self.parametric_estimator == False and self.cov:
            raise Exception("cov must be used for parametric estimation only")
        if self.parametric_estimator and self.cov == False:
            self.my_warning()
        
        self.x_name = self.x_names(x)
        self.y_name = self.y_names(y)
        self.x = self.format_x(x)
        self.y = self.format_y(y)
        rank_y = ss.rankdata(self.y, method='average')
        self.x_ones = np.c_[np.ones(self.x.shape[0]), self.x]
        
        if self.add_constant:
            self.x = self.format_x(x)
            self.y = self.format_y(y)
            self.x_ones = self.x
            x_no_ones = np.delete(self.x, 0, axis=1)
            self.x_name = self.x_names(x_no_ones)
        self.n, self.k = self.x_ones.shape

        # Estimator Gini non-parametric
        self.beta_coeff = self.gini_estimator(y, self.x_ones)
        self.residuals = self.y - self.x_ones@self.beta_coeff
                
        # Inference: non-parametric Gini estimator only (Jackknife over 1,000 iterations)
        if self.iter:
            if self.n < self.iter:
                self.iter = self.n
            if self.iter <= 1000 and self.iter > self.n:
                raise Exception("iterations must be > 1,000 and <= n.obs: default = 1,000 or n.obs"
                                )
            np.random.seed(123)
            x = self.x_ones[np.random.choice(self.n, self.iter, replace=False), :]
            y = self.y[np.random.choice(self.n, self.iter, replace=False)]
            n, k = x.shape
        else:
            x = self.x_ones 
            y = self.y
            n, k = x.shape
        beta_list = np.zeros((n,k)) 
        gini_R2_list = np.zeros((n,1))
        for i in range (n):
            x_jack = np.delete(x, i, axis=0)
            y_jack = np.delete(y, i, axis=0)
            beta_gini = self.gini_estimator(y_jack, x_jack)
            beta_list[i,:] = beta_gini.T
            gini_R2_list[i,:] = self.gini_R2(y_jack, x_jack)
        # Jackknife variance of coefficient estimates
        variance_coeff = (n-1)/n*(np.sum((beta_list - np.mean(beta_list, axis = 0))**2, axis =0))
        self.stdev_coeff = np.sqrt(variance_coeff)
        # Jackknife variance of Gini R-squared
        variance_gini_R2 = (n-1)/n*(np.sum((gini_R2_list - np.mean(gini_R2_list, axis = 0))**2, axis =0))
        self.stdev_gini_R2 = np.sqrt(variance_gini_R2)

        # Estimator Gini parametric 
        if self.parametric_estimator:
            self.beta_coeff = self.gini_estimator_parametric(self.y, self.x)
            self.residuals = self.y - self.x_ones@self.beta_coeff

        # Variance-covariance estimators
        if self.cov == 'instrument':
            rank_x = self.ranks(self.x_ones)
            sigma_residuals = np.sqrt(np.sum(self.residuals**2)/self.n)
            variance_coeff = sigma_residuals**2 * inv(rank_x.T@self.x_ones) @ (rank_x.T@rank_x) @ inv(self.x_ones.T@rank_x)
            self.stdev_coeff = np.sqrt(np.diag(variance_coeff))

        if self.cov == 'asympt':
            sigma_residuals = np.sqrt(np.sum(self.residuals**2)/self.n)
            variance_coeff_asympt = sigma_residuals**2 * inv(self.x_ones.T@self.x_ones)
            self.stdev_coeff = np.sqrt(np.diag(variance_coeff_asympt))
            
        if self.cov == 'bandwidth iid': # see statsmodels for quantile regression
            q = 0.5 # corresponding to LAD in quantile regressions
            # Greene (2008, p.407) writes that Stata 6 uses this bandwidth:
            # h = 0.9 * np.std(e) / (nobs**0.2)
            # Instead, we calculate bandwidth as in Stata 12
            
            kernels = {}
            #kernels['biw'] = lambda u: 15. / 16 * (1 - u**2)**2 * np.where(np.abs(u) <= 1, 1, 0)
            #kernels['cos'] = lambda u: np.where(np.abs(u) <= .5, 1 + np.cos(2 * np.pi * u), 0)
            #kernels['epa'] = lambda u: 3. / 4 * (1-u**2) * np.where(np.abs(u) <= 1, 1, 0)
            kernels['gauss'] = norm.pdf
            kernel = kernels['gauss'] # change the kernel
            iqre = ss.scoreatpercentile(self.residuals, 75) - ss.scoreatpercentile(self.residuals, 25)
            h = self.hall_sheather(self.n, q, alpha=.05)
            h = min(np.std(self.y),iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))
            fhat0 = 1. / (self.n * h) * np.sum(kernel(self.residuals / h))
            d = np.where(self.residuals > 0, (q/fhat0)**2, ((1-q)/fhat0)**2)
            vcov = pinv(self.x_ones.T@self.x_ones) @ (self.x_ones.T * d[np.newaxis, :] @self.x_ones) @ pinv(self.x_ones.T@self.x_ones)           
            self.stdev_coeff = np.sqrt(np.diag(vcov))
        
        if self.cov == 'Jackknife': 
            jack_iter = input('Enter the number of iterations')
            self.jack_iter = int(jack_iter)
            if self.n < self.jack_iter:
                raise Exception("Sample size too low to perform Jackknife iterations")
            np.random.seed(123)
            x = self.x[np.random.choice(self.n, self.jack_iter, replace=False), :]
            y = self.y[np.random.choice(self.n, self.jack_iter, replace=False)]
            n, k = x.shape
            beta_list = np.zeros((n,k+1)) 
            gini_R2_list = np.zeros((n,1))
            for i in range (n):
                x_jack = np.delete(x, i, axis=0)
                y_jack = np.delete(y, i, axis=0)
                beta_gini = self.gini_estimator_parametric(y_jack, x_jack)
                beta_list[i,:] = beta_gini.T
                gini_R2_list[i,:] = self.gini_R2(y_jack, x_jack)
            # Jackknife variance of coefficient estimates
            variance_coeff = (n-1)/n*(np.sum((beta_list - np.mean(beta_list, axis = 0))**2, axis =0))
            self.stdev_coeff = np.sqrt(variance_coeff)
            # Jackknife variance of Gini R-squared
            variance_gini_R2 = (n-1)/n*(np.sum((gini_R2_list - np.mean(gini_R2_list, axis = 0))**2, axis =0))
            self.stdev_gini_R2 = np.sqrt(variance_gini_R2)

        
        #if self.cov == False:
        #    self.cov = 'Jackknife'

        # Tests beta coefficients
        self.u_statistics_coeff = self.beta_coeff / self.stdev_coeff            # Jackknife
        self.p_values_coeff = (1-norm.cdf(np.abs(self.u_statistics_coeff)))*2   # Multiply by 2 for two-tailed test

        # Gini R2
        rank_residuals = ss.rankdata(self.residuals, method='average')
        self.Gini_R2 = 1 - (np.cov(self.residuals.T,rank_residuals.T)[0][1]**2) / (np.cov(self.y.T,rank_y.T)[0][1]**2)

        # U-stats Gini R2 
        self.u_statistics_gini_R2 = self.Gini_R2 / self.stdev_gini_R2
        self.p_values_gini_R2 = (1-norm.cdf(np.abs(self.u_statistics_gini_R2)))   # One-tailed test
        
        # Confidence intervals 95%
        self.coeff_lower_bound = self.beta_coeff - 1.96 * self.stdev_coeff
        self.coeff_upper_bound = self.beta_coeff + 1.96 * self.stdev_coeff
        
        # No. of outliers
        self.number_outliers = self.outliers(self.x_ones)
        
        # Results VIF
        x_not_ones = np.delete(self.x_ones, 0, axis=1)
        _,k = x_not_ones.shape
        vif_gini = [self.variance_inflation_factor(x_not_ones, i) for i in range(k)]
        vif_gini.insert(0, '--')
        self.vif_gini = np.array(vif_gini, dtype=object)
        if self.x.ndim >= 2:
            vif_ols = [np.round(variance_inflation_factor(x_not_ones, i), 4) for i in range(k)]
            vif_ols.insert(0, '--')
            self.vif_ols = np.array(vif_ols, dtype=object)
    
    def predict(self, x_test):
        x_test = self.format_x(x_test)
        n,k = x_test.shape
        if len(self.beta_coeff) == k:  # add_constant is True 
            return x_test@self.beta_coeff
        else:
            x_test = np.c_[np.ones(n), x_test]
            return x_test@self.beta_coeff
    
    def outliers(self, x):
        outliers_variables = []
        for i in range (1,self.k):
            outliers_variables.append(grubbs.max_test_indices(self.x_ones[:,i], alpha = 0.1))
        if len(outliers_variables) > 0:
            return len(outliers_variables)
    
    def summary(self, get_coeff = False):
        self.get_coeff = get_coeff
        if self.get_coeff:
            return self.beta_coeff
        else:
        # First table        
            table_data = [
            ["Dep. Variable:", self.y_name[0], "Gini R-squared:", np.round(self.Gini_R2,4)],
            ["Model:", "Gini Regression", "Adj. Gini R-squared:", np.round(1-(1-self.Gini_R2)*(self.n-1)/(self.n-self.k-1),4) ],
            ["Method:", 
            "Parametric" if self.parametric_estimator else "Non Parametric", "U-stat (Gini R2):", np.round(self.u_statistics_gini_R2[0],4)],
            ["Jackknife iterations", self.jack_iter if self.cov == 'Jackknife' else self.iter if not self.cov else None, "Prob (U-stat):", np.round(self.p_values_gini_R2[0],4)],
            ["Date:", datetime.now().strftime("%a, %d %b %Y"), "No. Outliers in x:", self.number_outliers],
            ["Time:", datetime.now().strftime("%H:%M:%S"), "Df Residuals:", self.n-self.k-1],
            ["Covariance type:", self.cov, "No. Observations:", self.n],
            ]
            header = ["Gini Regression"]
            formatted_data = [
            [
                f"{row[0].ljust(20)}{str(row[1]).rjust(20)}{''.ljust(5)}{row[2].ljust(20)}{str(row[3]).rjust(20)}"
            ]
            for row in table_data
            ]
            # Header to center above all columns
            header_str = " ".join(header).center(85)  # Assuming the total width of the table is 80 characters
            separator_line = '=' * len(header_str)
            print(separator_line)
            print(header_str)
            print(separator_line)
            for row in formatted_data:
                print(*row)
            if self.n < 50:
                print("Warning /!\ U-stats not accurate: too low number of observations")
            print(separator_line)
        
            # Second table
            coef_names = self.x_name
            coef_names.insert(0, 'Intercept')
            coef_data = [coef_names,
                        np.round(self.beta_coeff,4), 
                        np.round(self.stdev_coeff,4), 
                        np.round(self.u_statistics_coeff,4), 
                        np.round(self.p_values_coeff,4),
                        np.round(self.coeff_lower_bound,4),
                        np.round(self.coeff_upper_bound,4),
                        self.vif_gini,
                        self.vif_ols
                        ]
            coef_table_data = np.array(coef_data, dtype = object).T
            coef_table = SimpleTable(
                        coef_table_data,
                        headers=[' Variables ', ' Coeff ', ' Std err ', ' U ', 
                                ' P>|U| ', ' [0.025 ', ' 0.975] ',' VIF Gini', ' VIF OLS '
                        ], 
                        title=" Coefficients * and * Tests "
                        )
            print(coef_table)

     
     
class Hetero(GiniRegression):
    
    '''
    Gini estimator with Heteroskedasticity
    ----------
    Functions:
    ----------
    fit :
        - cov = 'WLS' : WLS Gini like covariance estimator
        - cov = 'iterative WLS' : WLS with convergence 200 iterations tol 1e-5 
        - cov = 'Breusch-Pagan' : covariance estimator 
    '''

    def __init__(self, hetero):
        super().__init__()
        self.hetero = hetero
        
    def hetero_FGGR(self, y, x):
        x = self.format_x(x)
        y = self.format_y(y)
        n,_ = x.shape
        x_ones = np.c_[np.ones(n), x]
        beta_coeff = self.gini_estimator(y, x_ones)
        # Step 1: Gets residuals
        residuals = y - x_ones@beta_coeff
        # Step 2: Brush Pagan estimations with Gini regression
        beta_coeff = self.gini_estimator(np.log(residuals ** 2), x_ones)
        omega = np.exp(x_ones@beta_coeff)
        P = np.diag(np.power(omega, -0.5)) # P = omega^{-0.5}
        return P  
     
    def hetero_wls(self, y, x):
        x = self.format_x(x)
        y = self.format_y(y)
        n,_ = x.shape
        x_ones = np.c_[np.ones(n), x]
        beta_coeff = self.gini_estimator(y, x_ones)        
        residuals = y - x_ones@beta_coeff
        omega = np.power(residuals, 2)/n
        P = np.diag(np.power(omega, -0.5)) # P = omega^{-0.5}
        return P, residuals**2
        
    def hetero_iterative_wls(self, y, x, max_iter = 100, tol = 1e-5):
        x = self.format_x(x)
        y = self.format_y(y)
        n,_ = x.shape
        x_ones = np.c_[np.ones(n), x]
        beta_coeff = self.gini_estimator(y, x_ones)
        error_covariance = self.hetero_wls(y, x)[1]
        for i in range(max_iter):
            # Step 1: compute Gini WLS
            weighted_x = x_ones / np.sqrt(error_covariance)[:, np.newaxis]
            weighted_y = y / np.sqrt(error_covariance)
            new_beta = self.gini_estimator(weighted_y, weighted_x)                
            # Step 2: update error covariance
            residuals = y - x_ones @ new_beta
            new_error_covariance = np.diag(residuals**2)               
            P = np.diag(np.power(residuals**2, -0.5))     
            # Step 3: check convergence
            if np.all(np.abs(new_beta - beta_coeff) < tol):
                break
            beta_coeff = new_beta
            error_covariance = new_error_covariance
            return P
            
    def fit(self, y, x, hetero = False):
        self.hetero = hetero
        self.cov = self.hetero
        x = self.format_x(x)
        y = self.format_y(y)
        n,_ = x.shape
        x_ones = np.c_[np.ones(n), x]
        if self.hetero == 'WLS':
            P = self.hetero_wls(y, x)[0]         
            super().fit(P@y, P@x_ones, add_constant = True)
        elif self.hetero == 'FGGR':
            P = self.hetero_FGGR(y, x)          
            super().fit(P@y, P@x_ones, add_constant = True)
        elif self.hetero == 'FGGR_IV':
            P = self.hetero_FGGR(y, x)
            x_hat = P @ x_ones
            super().fit(y, x_hat, IV = x_ones, add_constant = True)
        elif self.hetero == 'iterative WLS':
            P = self.hetero_iterative_wls(y, x, max_iter = 200, tol = 1e-5)        
            super().fit(P@y, P@x_ones, add_constant = True)
        else:
            raise Exception("Choose an heteroskedastic method: hetero = 'WLS', 'Breusch-Pagan' or 'iterative WLS' ")

    #def summary(self):
    #    self.cov = self.hetero
    #    super().summary()    
        
        
class Autocorrelation(GiniRegression):
    
    '''
    Gini estimator with MA(1)
    ----------
    Functions:
    ----------
    fit :
        - autocorrel == 'Prais-Winsten' : AR(1) estimator
        - correlogram == True : plot Gini-correlogram of residuals with Ljung_box test
    '''

    def __init__(self, autocorrel):
        super().__init__()
        self.autocorrel = autocorrel

    def autocorrel_gini(self, x, k):
        """
        Calculate the Gini autocorrelation of order k
        """
        r_x = ss.rankdata(x, method='average') / len(x)
        if k == 0:
            return 1
        else:
            n = np.sum((x[k:] - np.mean(x)) * (r_x[:-k] - np.mean(r_x)))
            d = np.sum((x - np.mean(x)) * (r_x - np.mean(r_x)))
        if n/d == 0:
            return np.nan 
        return n/d
        
    def autocorrel_gini_toeplitz_1(self, x):
        n = len(x)
        phi = self.autocorrel_gini(x, 1) 
        P = np.eye(n)
        P[0, 0] = np.sqrt(1 - phi**2)  
        for i in range(1, n):
            P[i, i - 1] = -phi     
        return P
    
    def autocorrel_gini_toeplitz_2(self, x):
        n = len(x)
        phi = self.autocorrel_gini(x, 1) 
        c = np.zeros((n,1))
        for i in range(n):
            c[i] = phi**i
        Omega = la.toeplitz(c)
        eigenvalues, eigenvectors = np.linalg.eigh(Omega)
        P = eigenvectors @ np.diag(eigenvalues**(-0.5)) @ eigenvectors.T
        return P
        
    def fit(self, y, x, residuals, autocorrel = False, correlogram = False):
        self.autocorrel = autocorrel
        #self.cov = self.autocorrel
        x = self.format_x(x)
        y = self.format_y(y)
        n,_ = x.shape
        x_ones = np.c_[np.ones(n), x]
        if self.autocorrel == 'Prais-Winsten':
            P = self.autocorrel_gini_toeplitz_1(residuals)
            super().fit(P@y, P@x_ones, add_constant = True)
        else: 
            P = self.autocorrel_gini_toeplitz_2(residuals)    
            super().fit(P@y, P@x_ones, add_constant = True)
        if correlogram == True:
            list_acf = []
            for i in range(0,12):
                list_acf.append(self.autocorrel_gini(residuals, i))
            n = len(residuals)
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(0,12), acf(residuals, nlags = 11), label='ACF')
            plt.plot(np.arange(0,12), list_acf, label='Gini ACF')
            plt.title('Autocorrelation functions')
            plt.xlabel('Lags')
            plt.ylabel('ACF values')
            plt.legend()
            plt.show()
            lags = np.arange(1,12)
            sacf2 = np.array(list_acf[1:12]) ** 2 / (n - np.arange(1, 11 + 1))
            qljungbox = n * (n + 2) * np.cumsum(sacf2)[lags - 1]
            adj_lags = lags 
            pval = np.full_like(qljungbox, np.nan)
            pval = stats.chi2.sf(qljungbox, adj_lags)
            Table = pd.DataFrame({"lb_stat": qljungbox, "lb_pvalue": pval},
                                    index=lags)
            print("Gini Autocorrelation:")
            print("--------------------------")
            print(Table)
            print("--------------------------")
            # Ljung-Box test
            ljung_box_result = acorr_ljungbox(residuals, lags=[11], return_df=True)
            Q_stat = ljung_box_result['lb_stat'].values  # Q-statistic values
            p_values = ljung_box_result['lb_pvalue'].values  # p-values for the Q-statistics
            Table_2 = pd.DataFrame({"lb_stat": Q_stat, "lb_pvalue": p_values},
                                    index=lags) 
            print("Autocorrelation:")
            print("--------------------------")
            print(Table_2)
            print("--------------------------")
                   
        return self.beta_coeff

    def summary(self):
        #self.cov = self.hetero
        super().summary()    
                                