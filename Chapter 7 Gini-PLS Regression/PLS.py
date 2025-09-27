from datetime import datetime
from statsmodels.iolib.table import SimpleTable
import pandas as pd
import numpy as np
from scipy import stats as ss
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from numpy.linalg import inv
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor 
try:
    from OUTLIERS import smirnov_grubbs as grubbs
except ImportError:
    from outliers import smirnov_grubbs as grubbs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class GiniPLS:
    
    '''
    Gini Regression
    Cite : Mussard, S., & Souissi-Benrejab, F. (2019). Gini-PLS Regressions. Journal of Quantitative Economics.
    => Modifications: the GAUSS code of the 2019 paper relied on rankindex that does not handle ties for rank computations.
    
    Variables : n_components

    ----------
    Functions:
    ----------
    fit :
        - method = 'Gini1-PLS1' algorithm based on the rank vectors in the partial least squares to derive the weights W
        - method = 'Gini2-PLS1' algorithm based on beta coeff of Gini regressions to derive the weights W
        - 
    '''

    def __init__(self):
        self = self

    def format_data(self, data):
        if isinstance(data, pd.DataFrame):
            return data.to_numpy().astype('float64')  
        elif isinstance(data, pd.Series):
            return data.to_numpy().astype('float64') 
        elif isinstance(data, np.ndarray):
            return data.astype('float64')
        else:
            raise ValueError("Input must be a pandas DataFrame or a NumPy array")

    def split_y_x(self, data):
        data = self.format_data(data)
        y = data[:, 0]
        X = data[:, 1:]
        return y, X

    def center_data(self, X, y):
        X_centered = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1) 
        y_centered = (y - np.mean(y)) / np.std(y, ddof=1)
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0, ddof=1)
        self.y_std = np.std(y, ddof=1)
        self.y_mean = np.mean(y)
        return X_centered, y_centered

    def ranks(self, x):
        if x.ndim == 1:
            rank = ss.rankdata(x, method='average')
        else:
            n, k = x.shape
            rank = np.zeros_like(x)
            for i in range(k):
                rank[:, i] = ss.rankdata(x[:, i], method='average')
        return rank
    
    def vif(self,x):
        VIF = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        return np.array(VIF)
          
    def gini_estimator(self, y, x): # x must contain the column for the constant
        rank_x = self.ranks(x)
        beta_gini = np.cov(y.T, rank_x.T)[0][1] / np.cov(x.T, rank_x.T)[0][1]
        return beta_gini
    
    def weights(self, y, X):
        cov_matrix = np.zeros((k,))
        n,k = X.shape
        for i in range(k):
            cov_matrix[i] = np.cov(y.T, X[:,i].T)[0][1] 
        return cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
    
    def cross_validation(self, n_splits = 5, random_state=None):
        if random_state is None:
            random_state = 0
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        Q2 = np.zeros((self.n_components,))
        PRESS = np.zeros((n_splits,self.n_components+1))
        RSS = np.zeros((n_splits,self.n_components+1))
        len_train = 0 
        len_test = 0
        X, y = self.center_data(self.X, self.y)
        for fold, (train_index, test_index) in enumerate(kf.split(self.X_centered)):
            X_train, X_test = X[train_index], X[test_index]                
            y_train, y_test = y[train_index], y[test_index]
            len_train += len(y_train) 
            len_test += len(y_test) 
            component_matrix_test = np.zeros((X_test.shape[0],self.n_components))
            for h in range(self.n_components):
                resids_test = np.zeros((X_test.shape[0],X_test.shape[1]))
                result = self.fit(y_train, X_train, method = self.method, CV = True, n_components = h+1)
                model_coef, RSS_h, weights, x_loads = result[0], result[1], result[2][:,:h+1], result[3]
                if h == 0:
                    y_pred = X_test @ weights @ model_coef.T
                    component_matrix_test[:,0] = (X_test @ weights).flatten()
                    RSS[fold,h] = np.sum((self.y_centered - np.mean(self.y_centered))**2) 
                    RSS[fold,h+1] = RSS_h 
                    PRESS[fold,h] = np.sum((y_test - y_pred)**2) 
                else:
                    for k in range(X_test.shape[1]):
                        resids_test[:,k] = X_test[:,k] - component_matrix_test[:,:h] @ x_loads[k,:h].T
                    component_matrix_test[:,h] = (resids_test @ weights[:,h]).flatten()
                    y_pred = component_matrix_test[:,:h+1] @ model_coef
                    PRESS[fold,h] = np.sum((y_test - y_pred)**2) 
                    RSS[fold,h+1] = RSS_h  
        Q2[0] = 1 - (np.sum(PRESS[:,0], axis = 0)/len_test) / (np.sum(RSS[:,0], axis = 0)/(len(self.y_centered)*n_splits))
        for i in range(1,self.n_components):
            Q2[i] = 1 - (np.sum(PRESS[:,i])/len_test) / (np.sum(RSS[:,i])/len_train)
        self.Q2 = Q2
        return print("Q2 statistics:", np.round(Q2, 4))

    def fit(self, y, X, method = True, CV = None, n_components = True):
        if CV is None:
            self.X_names = X
        y = self.format_data(y)
        X = self.format_data(X)
        X_centered, y_centered = self.center_data(X, y)
        X_centered_copy = X_centered
        components = np.zeros((len(X_centered), n_components)) # Gini-PLS components 
        n, k = X.shape
        rank_X = self.ranks(X_centered)
        rank_centered = rank_X - rank_X.mean(axis=0)
        weights = np.zeros((k,n_components+1))   # Weights: to built components based on ranks
        resids = np.zeros((n,n_components))      # Residuals of OLS regression between y and components
        beta_coeff = np.zeros((1,n_components))  # Beta coeff of OLS regression of y on components: c_1, c_2, etc.
        cov_matrix = np.zeros((k,))              # Covariance between resids and ranks
        x_loads = np.zeros((k,n_components))
           
        if method == 'Gini1-PLS1':
            # Weight w_1 and norm:
            for i in range(k):
                cov_matrix[i] = np.cov(y_centered.T, rank_X[:,i].T)[0][1] 
            weights[:,0] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
            
            # Components T, weights and norm
            for h in range(n_components):
                cov_matrix = np.zeros((k,))
                partial_resids = np.zeros((n,k))
                components[:,h] = X_centered_copy @ weights[:,h] 
                model = sm.OLS(y_centered, components[:,:h+1], hasconst=False).fit()
                model_coef = model.params
                resids[:,h] = model.resid 
                RSS = np.sum(model.resid**2)
                for i in range(k):
                    model = sm.OLS(rank_centered[:,i], components[:,:h+1], hasconst=False).fit()
                    partial_resids[:,i] = rank_centered[:,i] - components[:,:h+1] @ model.params.T
                    x_loads[i,h] = model.params[h]
                self.rank_partial_resids = self.ranks(partial_resids)
                for i in range(k):
                    cov_matrix[i] = np.cov(resids[:,h].T, self.rank_partial_resids[:,i].T, ddof=0)[0][1] 
                weights[:,h+1] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
                X_centered_copy = partial_resids
             
        if method == 'Gini2-PLS1':
            # Weight w_1:
            for i in range(k):
                cov_matrix[i] = self.gini_estimator(y_centered, X_centered[:,i]) 
            weights[:,0] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
            # Components T
            for h in range(n_components):
                cov_matrix = np.zeros((k,))
                partial_resids = np.zeros((n,k))
                components[:,h] = X_centered_copy @ weights[:,h] 
                model = sm.OLS(y_centered, components[:,:h+1], hasconst=False).fit()
                model_coef = model.params
                resids[:,h] = model.resid 
                RSS = np.sum(model.resid**2)
                for i in range(k):
                    model = sm.OLS(X_centered[:,i], components[:,:h+1], hasconst=False).fit()   
                    partial_resids[:,i] = model.resid
                    x_loads[i,h] = model.params[h]
                for i in range(k):
                    cov_matrix[i] = self.gini_estimator(resids[:,h], partial_resids[:,i])
                weights[:,h+1] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
                X_centered_copy = partial_resids

        if method == 'Gini3-PLS1':
            # Weight w_1 and norm:
            for i in range(k):
                cov_matrix[i] = np.cov(y_centered.T, rank_X[:,i].T)[0][1] 
            weights[:,0] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
            
            # Components T, weights and norm
            for h in range(n_components):
                cov_matrix = np.zeros((k,))
                partial_resids = np.zeros((n,k))
                components[:,h] = X_centered_copy @ weights[:,h] 
                model = sm.OLS(y_centered, components[:,:h+1], hasconst=False).fit()
                model_coef = model.params
                resids[:,h] = model.resid 
                RSS = np.sum(model.resid**2)
                for i in range(k):
                    model = sm.OLS(X_centered[:,i], components[:,:h+1], hasconst=False).fit()
                    partial_resids[:,i] = model.resid
                    x_loads[i,h] = model.params[h]
                for i in range(k):
                    cov_matrix[i] = np.cov(resids[:,h].T, partial_resids[:,i].T)[0][1]
                weights[:,h+1] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
                X_centered_copy = partial_resids

        if method ==  'PLS':
            # Weight w_1:
            for i in range(k):
                cov_matrix[i] = np.cov(y_centered.T, X_centered[:,i].T)[0][1] 
            weights[:,0] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
            # Components T
            RSS = 0
            for h in range(n_components):
                cov_matrix = np.zeros((k,))
                partial_resids = np.zeros((n,k))
                components[:,h] = X_centered_copy @ weights[:,h] 
                model = sm.OLS(y_centered, components[:,:h+1], hasconst=False).fit()
                model_coef = model.params
                resids[:,h] = model.resid
                RSS = np.sum(model.resid**2)
                for i in range(k):
                    model = sm.OLS(X_centered[:,i], components[:,:h+1], hasconst=False).fit()
                    partial_resids[:,i] = model.resid
                    x_loads[i,h] = model.params[h]
                for i in range(k):
                    cov_matrix[i] = np.cov(resids[:,h].T, partial_resids[:,i].T)[0][1]
                weights[:,h+1] = cov_matrix / np.sqrt(np.sum(cov_matrix*cov_matrix))
                X_centered_copy = partial_resids
                
        # CV is not None
        if CV is True:
            self.CV = CV 
            return model_coef, RSS, weights, x_loads, components, partial_resids
        else:
            # Setting variables for results of Gini-PLS (summary)
            model = sm.OLS(y_centered, components[:,:n_components+1], hasconst=False).fit()
            self.CV = None
            self.method = method
            self.model = model
            self.n_components = n_components
            self.method = method
            self.X = X
            self.y = y
            self.X_centered = X_centered
            self.y_centered = y_centered
            self.r_squared = model.rsquared
            self.beta_coeff = self.model.params
            self.std_errors = model.bse
            self.t_values = model.tvalues
            self.p_values_coeff = model.pvalues
            self.components = components
            self.weights = weights
            self.coeff_lower_bound = self.beta_coeff - 1.96 * self.std_errors
            self.coeff_upper_bound = self.beta_coeff + 1.96 * self.std_errors
            self.number_outliers = self.outliers(self.X_centered)[1]
            self.outliers_by_features = self.outliers(self.X_centered)[0]
            
            # X_loads beta coeff of partial regressions
            variable_names = self.x_names(self.X_names)
            component_names = self.t_names(self.components)
            self.x_loads_copy = x_loads
            self.x_loads = pd.DataFrame(x_loads, index=variable_names, columns=component_names)

            # Rd(R(X),t): Redundancy of rank X (or X) on components t 
            names = self.x_names(self.X_names)
            names.append('y')
            y = self.y_centered[:, np.newaxis]
            X_y = np.concatenate((self.X_centered, y), axis = 1)
            Rdx = np.zeros((X_centered.shape[1]+1, self.n_components))
            for comp in range(self.n_components):
                for i in range(X_centered.shape[1]+1):
                    if self.method == 'PLS':
                        Rdx[i,comp] = (np.corrcoef(self.components[:,comp].T, X_y[:,i].T)[0][1])**2                     
                    else: 
                        Rdx[i,comp] = (np.corrcoef(self.components[:,comp].T, X_y[:,i].T)[0][1])**2 
            Rdx = np.cumsum(Rdx, axis = 1)
            self.Rdx = pd.DataFrame(Rdx, index=names, columns=self.t_names(self.components))
            
            # VIF
            self.VIF = self.vif(self.X_centered)
            
            # Tests resids:
            self.dw_stat = durbin_watson(self.model.resid)
            exog = sm.add_constant(self.model.model.exog)
            _, pval, _, f_pval = het_breuschpagan(self.model.resid, exog)
            self.p_val = pval

            # VIP: derived from Matlab BSD licence
            beta = self.beta_coeff[:, np.newaxis]
            _, k = self.X_centered.shape
            _, h = self.components.shape
            vips = np.zeros((k,))
            s = np.diag(self.components.T @ self.components @ beta @ beta.T).reshape(h, -1)
            s = s.flatten()
            for i in range(k):
                weight = np.array([ (self.weights[i,j] / np.linalg.norm(self.weights[:,j]))**2 for j in range(h) ])
                vips[i] = np.sqrt(k*(s.T @ weight)/np.sum(s))
            self.vips = vips
        
    def outliers(self, x):
        outliers_numbers = []
        for i in range (x.shape[1]):
            outliers_numbers.append(len(grubbs.max_test_indices(x[:,i], alpha = 0.1)))
        outliers_numbers = np.array(outliers_numbers)
        return outliers_numbers, outliers_numbers.sum()
    
    def predict(self, x):
        result = self.coeff_reconstructed()
        coef_, intercept_ = result[0], result[1]
        x = self.format_data(x)
        x -= self.x_mean
        x /= self.x_std
        y_pred = x @ coef_.T + intercept_
        print("Predicted values in the initial space:", "\n", np.round(y_pred.ravel(),4),"\n")
        return y_pred.ravel()
    
    def coeff_reconstructed(self): # coefficients of the regression in the initial space
        self.x_rotations_ = np.dot(
        self.weights[:,0:self.n_components],
        pinv(np.dot(self.x_loads.T, self.weights[:,0:self.n_components])),
        )
        self.coef_ = np.dot(self.x_rotations_, self.beta_coeff.T)
        self.coef_ = (self.coef_ * self.y_std).T
        self.intercept_ = self.y_mean
        return self.coef_, self.intercept_
           
    def t_names(self, x):
        return [f"t{i+1}" for i in range(x.shape[1])]
    
    def x_names(self, x):
        if isinstance(x, pd.DataFrame):
            if x.columns.names:
                return list(x.columns) 
        else:
            return [f"x{i+1}" for i in range(x.shape[1])]

    def summary(self):
        #coefficients in the initial space
        self.coeff_reconstructed()
        
        #Weights
        names_comp = self.t_names(self.components)
        self.weight_df = pd.DataFrame(self.weights[:,:self.n_components], index=self.x_names(self.X_names), columns=names_comp)

        table_data = [
            ["Dep. Variable:", "y", "R-squared:", np.round(self.r_squared,4)],
            ["Model:", "Gini PLS Regression", "Df Residuals:", self.X_centered.shape[0]-self.n_components-1],
            ["Method:", 
            "Gini1-PLS1" if self.method == 'Gini1-PLS1' else
            "Gini2-PLS1" if self.method == 'Gini2-PLS1' else
            "Gini3-PLS1" if self.method == 'Gini3-PLS1' else "PLS", "No. Outliers in x:", self.number_outliers],
            ["Date:", datetime.now().strftime("%a, %d %b %Y"), "Time:", datetime.now().strftime("%H:%M:%S")],
            ["Breusch-Pagan p-val:", np.round(self.p_val,4), "Durbin Watson:", np.round(self.dw_stat,4)]
            ]
        header = ["PLS Gini Regression"]
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
        print(separator_line)
            
        headers=[' Variables ', ' Coeff ', ' Std err ', ' t ', 
                            ' P>|t| ', ' [0.025 ', ' 0.975] ', ' Q2 '
                ]
        header = ["Regression {}".format(self.method)] 
        header_str = " ".join(header).center(71)  # Assuming the total width of the table is 80 characters
        separator_line = '=' * len(header_str)
        print(separator_line)
        print(header_str)
        coef_names = self.t_names(self.components)
        coef_data = [coef_names,
                    np.round(self.beta_coeff,4), 
                    np.round(self.std_errors,4), 
                    np.round(self.t_values,4), 
                    np.round(self.p_values_coeff,4),
                    np.round(self.coeff_lower_bound,4),
                    np.round(self.coeff_upper_bound,4)
                    ]
        coef_table_data = np.array(coef_data, dtype = object).T
        coef_table = SimpleTable(
                    coef_table_data, headers=headers,
                    )
        print(coef_table)
        
        data = {
            'Variables': self.x_names(self.X_names),  
            'Coefficients': np.round(self.coef_,4),
            'VIP': np.round(self.vips,4),
            'VIF': np.round(self.VIF,4),
            'Outliers': self.outliers_by_features
        }
        df = pd.DataFrame(data)
        new_row_df = pd.DataFrame({'Coefficients': [self.intercept_]}, index=[1])
        df = pd.concat([new_row_df, df])
        column_order = ['Variables', 'Coefficients', 'VIP', 'VIF', 'Outliers']
        df = df[column_order]
        df.iloc[0,0] = 'Intercept'
        df.iloc[:, 2] = df.iloc[:, 2].astype(str)
        df.iloc[0, 2] = '--'
        df.iloc[:, 3] = df.iloc[:, 3].astype(str)
        df.iloc[0, 3] = '--'
        df.iloc[:, 4] = df.iloc[:, 4].astype(str)
        df.iloc[0, 4] = '--'
                
        separator_line_2 = '-' * 62
        print("                 Stats on Features")
        print(separator_line_2)
        print(df)
        print(separator_line_2)
        print("Redundancy on X or R(x): R2")
        print(self.Rdx)
        print(separator_line_2)
        print("Weights")
        print(self.weight_df)
        print(separator_line_2)
        print("X_loads: beta coeff of PLS")
        print(self.x_loads)
        print("\n")

    def get_index_names(self, df):
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.RangeIndex):
                return list(range(1, len(df) + 1))
            else:
                return df.index.tolist()
        elif isinstance(df, np.ndarray):
            return list(range(1, len(df) + 1))
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy ndarray.")

    def plot(self):
        names = self.get_index_names(self.X_names)
        if self.n_components == 1:
            print("Please use at least 2 components for 2D and 3D diagrams")
        elif self.n_components == 2:
            plt.scatter(self.components[:,0], self.components[:,1])
            for i, name in enumerate(names):
                plt.text(self.components[i, 0], self.components[i, 1] - 0.05, name, ha='right', va='top')
            plt.title('Projected points')
            plt.xlabel('1st component')
            plt.ylabel('2d component')
            plt.show()
        else:    
            fig = plt.figure(1, figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.components[:, 0], self.components[:, 1], self.components[:, 2], c=self.components[:, 0],
                       cmap=plt.cm.Set1, edgecolor='k', s=40)
            ax.set_title("Projected points")
            ax.set_xlabel("1st component")
            ax.xaxis.set_ticklabels([])
            ax.set_ylabel("2nd component")
            ax.yaxis.set_ticklabels([])
            ax.set_zlabel("3rd component")
            ax.zaxis.set_ticklabels([])
            ax.view_init(elev=30, azim=45)
            for i, name in enumerate(names):
                ax.text(self.components[i, 0], self.components[i, 1], self.components[i, 2], name, size=10, zorder=1, color='k')
            plt.show()
        #Graph regression line     
        y_max = self.y_centered.max()
        y_min = self.y_centered.min()
        ax = sns.scatterplot(x=self.model.fittedvalues, y=self.y_centered)
        ax.set(ylim=(y_min, y_max))
        ax.set(xlim=(y_min, y_max))
        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Actual values")
        x_ref = y_ref = np.linspace(y_min, y_max, 100)
        plt.plot(x_ref, y_ref, color='red', linewidth=1)
        plt.title("Actual vs. Predicted (in the latent space)")
        plt.show()
        # Circle of correlations
        if self.n_components >= 2:
            names = self.x_names(self.X_names)
            names.append('y')
            X = np.hstack([self.X_centered, self.y_centered[:, np.newaxis]])
            correl_y = np.zeros((X.shape[1], 2))
            for k in range(X.shape[1]):
                for comp in range(2):
                    correl_y[k,comp] = np.corrcoef(X[:,k].T, self.components[:,comp].T)[0][1]
            fig = plt.figure(figsize=(5.5,5.5))
            ax = fig.add_subplot(1, 1, 1)
            for i, j, label in zip(correl_y[:,0],correl_y[:,1], names):
                if label == 'y':
                    plt.text(i+0.04, j, label, color='red')
                    plt.arrow(0, 0, i, j, color='red', head_width=0.02, head_length=0.02)
                else:
                    plt.text(i+0.04, j, label, color='black')
                    plt.arrow(0, 0, i, j, color='gray', head_width=0.02, head_length=0.02)
            plt.axis((-1,1,1,-1))
            # Circle
            circle = plt.Circle((0,0), radius=0.992, color='blue', fill=False)
            ax.add_patch(circle)
            plt.axvline(0, color='k') # vertical axis
            plt.axhline(0, color='k') # horizontal axis
            plt.xlabel("Component 1 (t1)")
            plt.ylabel("Component 2 (t2)")
            plt.title("Circle of correlation")
            plt.show()

    def summary_OLS(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y, X).fit()
        return print(model.summary())

