from datetime import datetime
from scipy.stats import norm
from statsmodels.iolib.table import SimpleTable
from statsmodels.compat.python import lrange
import torch
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import statsmodels.api as sm
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
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings

class GiniFixedEffects(object):
    
    '''
    Gini estimator for fixed effetcs panel data
    The data must be sorted along groups
    ----------
    Functions:
    ----------
    fit :
        - 
        - 
    '''
    
    def __init__(self, parametric_estimator = False, cov = False, iter = False):
        self.parametric_estimator = parametric_estimator
        self.cov = cov
        self.iter = iter

    def format_x(self, x):
        if isinstance(x, pd.DataFrame):
            return x.values  
        elif isinstance(x, torch.Tensor):
            return x.to_numpy()
        else:
            return x
    
    def format_y(self, y):
        if isinstance(y, pd.DataFrame):
            y = y.values
            return y.flatten() 
        elif isinstance(y, torch.Tensor):
            return y.to_numpy().flatten()
        elif isinstance(y, pd.Series):
            return y.to_numpy()
        else:
            return y.flatten()
            
    def ranks(self, x):
        x = self.format_x(x)
        n, k = x.shape
        rank = np.zeros_like(x)
        for i in range(k):
            rank[:,i] = ss.rankdata(x[:,i], method='average')
        return rank
    
    def vif(self,x):
        x = self.format_x(x)
        VIF = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        return np.array(VIF)
    
    def outliers(self, x):
        outliers_numbers = []
        for i in range (len(x.columns)):
            outliers_numbers.append(len(grubbs.max_test_indices(x.iloc[:,i].values, alpha = 0.1)))
        outliers_numbers = np.array(outliers_numbers)
        return outliers_numbers, outliers_numbers.sum()
        
    def gini_estimator_parametric(self, y, x):
        x = self.format_x(x)
        y = self.format_y(y)
        y = y.reshape(-1, 1)
        n,_ = x.shape
        X = np.concatenate((y, x), axis = 1) 
        def gini_param(coef, X_data):
            residuals = X_data[:, 0] - (coef * X_data[:, 1:]).sum(axis=1)
            rank_residuals = ss.rankdata(residuals, method='average')
            return 1/n*np.cov(residuals.T,rank_residuals.T)[0][1] # function to minimize
        # Output of gini_param
        initial = [0] * (x.shape[1])
        result = minimize(gini_param, initial, args=(X,))
        beta_gini_parametric = result.x
        return beta_gini_parametric
    
    # Stratified sampling for jackknife inference in case of large samples
    def sampling_within_groups(self, df, id_group, min_obs):
        sampled_groups = []
        group_sizes = []
        unique_groups = df[id_group].unique()
        for group in unique_groups:
            group_df = df[df[id_group] == group]
            sample_size = min(min_obs, group_df.shape[0])
            sampled_group = group_df.sample(n=sample_size)
            sampled_groups.append(sampled_group)
            group_sizes.append(sampled_group.shape[0])
            sampled_df = pd.concat(sampled_groups)
        if not all(size == group_sizes[0] for size in group_sizes):  
                raise ValueError("Not all groups have the same size.")        
        return sampled_df, group_sizes
    
    def within_group_transform(self, df, id_group, y, X):
        # Convert data
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='y')
        elif not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series or a NumPy array")
        y_name = y.name if y.name is not None else 'y'
        self.y_name = y_name
        
        if isinstance(X, np.ndarray):
            num_features = X.shape[1] if len(X.shape) > 1 else 1  # Number of columns/features
            X = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(num_features)])
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame or a NumPy array")
        X_names = X.columns.tolist()
        self.X_names = X_names

        # Compute group means
        group_means_y = df.groupby(id_group)[y_name].mean().reset_index(name=f'{y_name}_mean')
        group_means_X = df.groupby(id_group)[X_names].mean().reset_index()
        group_means_X.columns = [id_group] + [f'{name}_mean' for name in X_names]

        # Merge the individual means back to the original dataframe
        merged_df = df.merge(group_means_y, on=id_group, how='left')
        merged_df = merged_df.merge(group_means_X, on=id_group, how='left')

        # Subtract individual means to center the data
        merged_df[f'{self.y_name}_centered'] = merged_df[y_name] - merged_df[f'{self.y_name}_mean']
        for name in self.X_names:
            merged_df[f'{name}_centered'] = merged_df[name] - merged_df[f'{name}_mean']

        # Ranks X : rank mean and centering 
        ranked_X = pd.DataFrame(self.ranks(df[self.X_names].values), columns=X_names, index=df.index)
        group_means_ranked_X = ranked_X.groupby(df[id_group]).mean().reset_index()
        group_means_ranked_X.columns = [id_group] + [f'{name}_ranked_mean' for name in self.X_names]     
        merged_df = merged_df.merge(group_means_ranked_X, on=id_group, how='left')
        merged_df.index = ranked_X.index # align index to make difference between columns
        for name in X_names:
            ranked_col_name = f'{name}_ranked'
            rank_mean_col_name = f'{name}_ranked_mean'
            merged_df[f'{name}_ranked_centered'] = ranked_X[name] - merged_df[f'{name}_ranked_mean']
            #print(ranked_X[name], merged_df[rank_mean_col_name])
        return merged_df
    
    def between_group_transform(self, df, id_group, y, X):
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='y')
        elif not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series or a NumPy array")
        y_name = y.name if y.name is not None else 'y'
        self.y_name = y_name
        
        if isinstance(X, np.ndarray):
            num_features = X.shape[1] if len(X.shape) > 1 else 1
            X = pd.DataFrame(X, columns=[f'x{i + 1}' for i in range(num_features)])
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame or a NumPy array")
        X_names = X.columns.tolist()
        self.X_names = X_names
        
        # Group means and overall mean
        group_means_y = df.groupby(id_group)[self.y_name].mean().reset_index(name=f'{y_name}_mean')
        grand_mean_y = y.mean()
        group_means_X = df.groupby(id_group)[X_names].mean().reset_index()
        grand_means_X = df[self.X_names].mean()
        group_means_X.columns = [id_group] + [f'{name}_mean' for name in self.X_names]

        # Grand mean centering
        merged_df = df.merge(group_means_y, on=id_group, how='left')
        merged_df = merged_df.merge(group_means_X, on=id_group, how='left')
        merged_df[f'{self.y_name}_centered'] = merged_df[f'{self.y_name}_mean'] - grand_mean_y
        for name in self.X_names:
            merged_df[f'{name}_centered'] = merged_df[f'{name}_mean'] - grand_means_X[name]

        # Handle ranks with consideration for unique indexes
        ranked_X = pd.DataFrame(self.ranks(df[self.X_names].values), columns=self.X_names, index=df.index).reset_index(drop=True)
        grand_means_ranked_X = ranked_X.mean()
        for name in self.X_names:
            ranked_col_name = f'{name}_ranked_centered'
            merged_df[ranked_col_name] = ranked_X[name] - grand_means_ranked_X[name]
        return merged_df

    def compute_fixed_effects(self, df, id_group, y, beta_coeff):
        y_means = df.groupby(id_group)[self.y_name].mean()
        y_predicted_means = np.dot(df.groupby(id_group)[self.X_names].mean().values, beta_coeff)
        fixed_effects = y_means.values - y_predicted_means
        average_intercept = np.mean(fixed_effects)
        return fixed_effects, average_intercept
    
    def fit(self, df, id_group, y, X, method = True, cov = False, estimator = False, centering = False, sampling = None):
        '''
        => method : 'OLS' or 'Gini'
        => estimator : 'parametric' for parametric Gini regression. 
        Default is Gini non-parametric with Jackknife
        => centering : 'between' for between-group estimator. 
        Default is within-group estimator.
        => cov: 'instrument' for instrumental variables inference. 
        Default is Jackknife. Be carreful of time estimation ! Use sampling if the size of the data set is large.
        => sampling = int : take at random observations within groups to perform jackknife faster. It takes 20%
        of the sample within each group. If the size is no higher than 100, then 40% of the sample within each group is taken, etc. until 80%. 
        '''
        self.method = method
        self.cov = cov
        self.estimator = estimator
        self.X = X
        self.y = y
        self.id_group = id_group
        self.centering = centering
        self.sampling = sampling
        
        # Sampling
        if sampling:
            df_clean = df.dropna()
            sampled_df, self.group_sizes = self.sampling_within_groups(df_clean, id_group, min_obs=sampling)
            df = sampled_df.reset_index(drop=True)
            self.X = self.X.reset_index(drop=True).reindex(df.index)
            self.y = self.y.reset_index(drop=True).reindex(df.index)
            X = self.X
            y = self.y
        
        # Centering
        if self.centering == 'between':
            transformed_df = self.between_group_transform(df, id_group, y, X)
        else:
            transformed_df = self.within_group_transform(df, id_group, y, X)
            
        y_centered = transformed_df[f'{self.y_name}_centered']
        X_centered = transformed_df[[f'{var}_centered' for var in X]]
        X_ranked_centered = transformed_df[[f'{var}_ranked_centered' for var in X]]
        self.X_centered = X_centered
        n,k = self.X_centered.shape
        self.y_centered = y_centered
        self.nb_groups = df[id_group].nunique()
        self.n = n

        # OLS
        if method == 'OLS':
            self.model = sm.OLS(y_centered, X_centered).fit()
            n,k = X_centered.shape
            beta_coeff = self.model.params
            fixed_effects, intercept = self.compute_fixed_effects(df, id_group, y, beta_coeff)
            self.fixed_effects = fixed_effects
            self.intercept = intercept
            var_cov_matrix = np.sum(self.model.resid**2)/(n-k) * inv(X_centered.T @ X_centered)
            groups_unique = df[id_group].unique()
            fixed_effects_dict = dict(zip(groups_unique, fixed_effects))
            fixed_expanded = df[id_group].map(fixed_effects_dict)
            group_sizes = df.groupby(id_group).size()
            average_observations = group_sizes.mean()
            # Asymptotic test (Green Chapter 13, 5th edition)
            var_residuals = np.sum(self.model.resid**2)/(n-k) / average_observations
            var_fixed_effects = var_residuals + df.groupby(id_group)[self.X_names].mean().values @ var_cov_matrix @ df.groupby(id_group)[self.X_names].mean().values.T
            var_fixed_diag = np.diag(var_fixed_effects)
            T_stat = fixed_effects / np.sqrt(var_fixed_diag)
            self.p_values_T_stat = (1-norm.cdf(np.abs(T_stat)))*2 
            # Asymptotic test for the intercept
            ones = np.ones(len(self.fixed_effects))
            var_intercept = ones.T @ var_fixed_effects @ ones / n**2
            T_stat = self.intercept / np.sqrt(var_intercept)
            self.p_values_intercept = (1-norm.cdf(np.abs(T_stat)))*2 
            
        # Gini
        if method == 'Gini': 
            if self.estimator == 'parametric':
                self.beta_coeff = self.gini_estimator_parametric(y_centered, X_centered)
            else:
                self.beta_coeff = inv(X_ranked_centered.T @ X_centered) @ (X_ranked_centered.T @ y_centered)
            self.residuals = y_centered - X_centered @ self.beta_coeff
                        
            # Gini R2
            rank_y = ss.rankdata(y_centered, method='average')
            rank_residuals = ss.rankdata(self.residuals, method='average')
            self.Gini_R2 = 1 - (np.cov(self.residuals.T,rank_residuals.T)[0][1]**2) / (np.cov(y_centered.T,rank_y.T)[0][1]**2)
            
            # Fixed effects
            self.fixed_effects, self.intercept_ = self.compute_fixed_effects(df, id_group, y, self.beta_coeff)

            # Variance-covariance estimators
            if self.cov == 'instrument':
                sigma_residuals = np.sqrt(np.sum(self.residuals**2)/(self.n-k))
                variance_coeff = sigma_residuals**2 * inv(X_ranked_centered.T @ X_centered) @ (X_ranked_centered.T @ X_ranked_centered) @ inv(X_centered.T @ X_ranked_centered)
                self.st_dev_coeff = np.sqrt(np.diag(variance_coeff))
                groups_unique = df[id_group].unique()
                fixed_effects_dict = dict(zip(groups_unique, self.fixed_effects))
                fixed_expanded = df[id_group].map(fixed_effects_dict)
                group_sizes = df.groupby(id_group).size()
                average_observations = group_sizes.mean()
                # Asymptotic test (Green Chapter 13, 5th edition)
                var_fixed_effects = sigma_residuals**2/average_observations + df.groupby(id_group)[self.X_names].mean().values @ variance_coeff @ df.groupby(id_group)[self.X_names].mean().values.T
                T_stat = self.fixed_effects / np.sqrt(np.diag(var_fixed_effects))
                self.p_values_T_stat = (1-norm.cdf(np.abs(T_stat)))*2 
                # Asymptotic test for the intercept
                ones = np.ones(len(self.fixed_effects))
                var_intercept = ones.T @ var_fixed_effects @ ones / n**2
                T_stat = self.intercept_ / np.sqrt(var_intercept)
                self.p_values_intercept = (1-norm.cdf(np.abs(T_stat)))*2 
                
            else:
                # Jackknife
                beta_coeff_list = [] 
                fixed_effects_list = []
                intercept_list = []
                Gini_R2_list = []
                test_R2_list = []
                for i in range(n):
                    df_jack = df.drop(df.index[i])
                    y_jack = y.drop(y.index[i])
                    X_jack = X.drop(X.index[i])
                    transformed_df = self.within_group_transform(df_jack, id_group, y_jack , X_jack)
                    y_centered_jack = transformed_df[f'{self.y_name}_centered']
                    X_centered_jack = transformed_df[[f'{var}_centered' for var in X]]
                    X_ranked_centered_jack = transformed_df[[f'{var}_ranked_centered' for var in X]]
                    if self.estimator == 'parametric':
                        beta_coeff = self.gini_estimator_parametric(y_centered_jack, X_centered_jack)
                    else:
                        beta_coeff = inv(X_ranked_centered_jack.T @ X_centered_jack) @ (X_ranked_centered_jack.T @ y_centered_jack)  
                    beta_coeff_list.append(beta_coeff)
                    residuals = y_centered_jack - X_centered_jack @ beta_coeff
                    #R2
                    rank_y = ss.rankdata(y_centered_jack, method='average')
                    rank_residuals = ss.rankdata(residuals, method='average')
                    Gini_R2 = 1 - (np.cov(residuals.T,rank_residuals.T)[0][1]**2) / (np.cov(y_centered_jack.T,rank_y.T)[0][1]**2)
                    Gini_R2_list.append(Gini_R2)
                    #fixed effects
                    fixed_effects, average_intercept = self.compute_fixed_effects(df_jack, id_group, y_jack, beta_coeff)
                    fixed_effects_list.append(fixed_effects)
                    intercept_list.append(average_intercept)
                    #R2 with fixed effects: test
                    residuals_with_Feffects = y_jack - X_jack @ beta_coeff - fixed_effects.sum()
                    rank_residuals_with_Feffects = ss.rankdata(residuals_with_Feffects, method='average')
                    U_1 = np.cov(residuals_with_Feffects.T,rank_residuals_with_Feffects.T)[0][1]
                    residuals_without_Feffects = y_jack - X_jack @ beta_coeff - average_intercept
                    rank_residuals_without_Feffects = ss.rankdata(residuals_without_Feffects, method='average')
                    U_2 = np.cov(residuals_without_Feffects.T,rank_residuals_without_Feffects.T)[0][1]
                    test_R2_list.append(np.abs(U_1 - U_2))
                    
                Gini_R2 = np.array(Gini_R2_list)
                fixed_effects = np.array(fixed_effects_list)
                intercept = np.array(intercept_list) 
                beta_coeffs = np.array(beta_coeff_list)
                test_R2 = np.array(test_R2_list)
                
                # standard dev.
                self.st_dev_Gini_R2 = np.sqrt(((n - 1) / n) * np.sum((Gini_R2 - np.mean(Gini_R2, axis=0))**2, axis=0))
                self.st_dev_coeff = np.sqrt(((n - 1) / n) * np.sum((beta_coeffs - np.mean(beta_coeffs, axis=0))**2, axis=0))
                self.st_dev_fixed_effects = np.sqrt(((n - 1) / n) * np.sum((fixed_effects - np.mean(fixed_effects, axis=0))**2, axis=0))
                self.st_dev_intercept = np.sqrt(((n - 1) / n) * np.sum((intercept - np.mean(intercept, axis=0))**2, axis=0))
                self.st_dev_test_R2 = np.sqrt(((n - 1) / n) * np.sum((test_R2 - np.mean(test_R2, axis=0))**2, axis=0))
                
                # Test fixed effects with U-statistics
                residuals_with_Feffects = self.y - self.X @ self.beta_coeff - self.fixed_effects.sum() - self.intercept_
                residuals_without_Feffects = self.y - self.X @ self.beta_coeff - self.intercept_
                rank_residuals_with_Feffects = ss.rankdata(residuals_with_Feffects, method='average')
                rank_residuals_without_Feffects = ss.rankdata(residuals_without_Feffects, method='average')
                U_1 = np.cov(residuals_with_Feffects.T,rank_residuals_with_Feffects.T)[0][1]
                U_2 = np.cov(residuals_without_Feffects.T,rank_residuals_without_Feffects.T)[0][1]
                self.test_R2 = np.abs(U_1 - U_2)
            
                # Test fixed effects : p_values
                self.u_statistics_fixed = self.fixed_effects / self.st_dev_fixed_effects           # Jackknife
                self.p_values_fixed = (1-norm.cdf(np.abs(self.u_statistics_fixed)))*2   # Multiply by 2 for two-tailed test
                self.u_statistics_intercept = self.intercept_ / self.st_dev_intercept           # Jackknife
                self.p_values_intercept = (1-norm.cdf(np.abs(self.u_statistics_intercept)))*2   # Multiply by 2 for two-tailed test
                self.u_statistics_test_R2 = self.test_R2 / self.st_dev_test_R2
                self.p_values_test_R2 = (1-norm.cdf(np.abs(self.u_statistics_test_R2)))*2   # Multiply by 2 for two-tailed test
                
            # Tests beta coefficients
            self.u_statistics_coeff = self.beta_coeff / self.st_dev_coeff           # Jackknife
            self.p_values_coeff = (1-norm.cdf(np.abs(self.u_statistics_coeff)))*2   # Multiply by 2 for two-tailed test

            # U-stats Gini R2 
            if self.cov != 'instrument':
                self.u_statistics_gini_R2 = self.Gini_R2 / self.st_dev_Gini_R2
                self.p_values_gini_R2 = (1-norm.cdf(np.abs(self.u_statistics_gini_R2)))   # One-tailed test
            
            # Confidence intervals 95%
            self.coeff_lower_bound = self.beta_coeff - 1.96 * self.st_dev_coeff
            self.coeff_upper_bound = self.beta_coeff + 1.96 * self.st_dev_coeff
            
            # No. of outliers
            self.number_outliers = self.outliers(self.X_centered)[1]
            self.number_features = self.outliers(self.X_centered)[0]
            
            # Tests resids:
            self.dw_stat = durbin_watson(self.residuals)
            exog = sm.add_constant(X_centered)
            try:
                _, pval, _, f_pval = het_breuschpagan(self.residuals, exog)
                self.p_val = pval
            except Exception as e:
                print(f"An error occurred during the Breusch-Pagan test: {e}")
                self.p_val = None
              
        #VIF
        self.VIF = self.vif(self.X)

        # Group Names
        unique_id_groups = transformed_df[id_group].unique()
        self.unique_id_groups_list = list(unique_id_groups)
        
    def summary(self):
        if self.method == 'OLS':
            print(self.model.summary())
            # Fixed effects :
            data = {
                    'Groups': self.unique_id_groups_list,  
                    'Coefficients': np.round(self.fixed_effects,4),
                    'p_value': np.round(self.p_values_T_stat,4)
                    }
            df = pd.DataFrame(data)
            add_const = {'Groups': 'Intercept', 'Coefficients': np.round(self.intercept,4), 'p_value': self.p_values_intercept}
            new_row = pd.DataFrame(add_const, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
            separator_line_2 = '-' * 62
            print(separator_line_2)
            print("                 Fixed Effects")
            print(separator_line_2)
            print(df)
            print(separator_line_2)

        else:
            table_data = table_data = [
                ["Dep. Variable:", "y", "Gini R-squared:", np.round(self.Gini_R2,4)],
                ["Model:", "Fixed Effects", "Gini R2 (p_val):", np.round(self.p_values_gini_R2, 4) if hasattr(self, 'p_values_gini_R2') else 'NA'],
                ["Method:", 'Gini Non Parametric' if self.method == 'Gini' else 'Gini Parametric', "No. Outliers in x:", self.number_outliers],
                ["Date:", datetime.now().strftime("%a, %d %b %Y"), "Time:", datetime.now().strftime("%H:%M:%S")],
                ["Breusch-Pagan p-val:", np.round(self.p_val,4) if self.p_val is not None else None, "Durbin Watson:", np.round(self.dw_stat,4)],
                ["Centering:", "Between-group" if self.centering == 'between' else "Within-group", "Df Residuals:", self.n - self.X_centered.shape[1]-self.nb_groups-1],
                ]
            header = ["Gini Regression: Panel Data"]
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
            #print(separator_line)
            print(header_str)
            coef_names = self.X_names
            coef_data = [coef_names,
                        np.round(self.beta_coeff,4), 
                        np.round(self.st_dev_coeff,4), 
                        np.round(self.u_statistics_coeff,4), 
                        np.round(self.p_values_coeff,4),
                        np.round(self.coeff_lower_bound,4),
                        np.round(self.coeff_upper_bound,4)
                        ]
            coef_table_data = np.array(coef_data, dtype = object).T
            coef_table = SimpleTable(
                        coef_table_data, headers=headers,
                        )
            print(coef_table)

            if self.cov != 'instrument':
                intercept = np.array(self.intercept_)  
                fixed_table = np.append(self.fixed_effects, intercept) 
                self.unique_id_groups_list.append('Intercept')
                p_val = np.array(self.p_values_intercept)  
                add_constant = np.append(self.p_values_fixed, p_val) 
                separator_line_2 = '-' * 62
                data = {
                    'Groups': self.unique_id_groups_list,  
                    'Coefficients': np.round(fixed_table,4),
                    'p_values': np.round(add_constant, 4),
                }
                if self.sampling:
                    self.group_sizes.append('--')
                    data = {
                    'Groups': self.unique_id_groups_list,  
                    'Coefficients': np.round(fixed_table,4),
                    'p_values': np.round(add_constant, 4),
                    'Group sizes': self.group_sizes,
                    }
                df = pd.DataFrame(data)
                print("                 Fixed Effects")
                print(separator_line_2)
                print(df)
            else:
                data = {
                    'Groups': self.unique_id_groups_list,  
                    'Coefficients': np.round(self.fixed_effects,4),
                    'p_values': np.round(self.p_values_T_stat, 4),
                }
                df = pd.DataFrame(data)
                add_const = {'Groups': 'Intercept', 'Coefficients': np.round(self.intercept_,4), 'p_values': np.round(self.p_values_intercept,4)}
                new_row = pd.DataFrame(add_const, index=[0])
                df = pd.concat([df, new_row], axis = 0)
                print("                 Fixed Effects")
                separator_line_2 = '-' * 62
                print(separator_line_2)
                print(df)
                print(separator_line_2)

            print(separator_line_2)
            print("                 Stats on Features")
            print(separator_line_2)
            data = {
                'Features': self.X_names,  
                'VIF': np.round(self.VIF,4),
                'Outliers': self.number_features
            }
            df = pd.DataFrame(data)
            print(df)
            print(separator_line_2)
            if not hasattr(self, 'test_R2') and not hasattr(self, 'p_values_test_R2'):
                pass  
            else:
                if hasattr(self, 'test_R2') and hasattr(self, 'p_values_test_R2'):
                    print(separator_line_2)
                    print("             Test of individual heterogeneity")
                    print(separator_line_2)
                    data = {
                        'U-statistics': [np.round(self.test_R2,4)],
                        'p_value': [np.round(self.p_values_test_R2,4)],
                    }
                    df = pd.DataFrame(data)
                    print(df)
                    print(separator_line_2) 
#                   print("U-statistics:", self.test_R2, "\n", "p_value:", self.p_values_test_R2)

