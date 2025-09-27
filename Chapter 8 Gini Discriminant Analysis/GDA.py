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
try:
    from OUTLIERS import smirnov_grubbs as grubbs
except ImportError:
    from outliers import smirnov_grubbs as grubbs
from sklearn.metrics import classification_report  
from sklearn.model_selection import KFold


class Model(object):
    def __init__(self):
        pass
    def _preprocess(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        assert isinstance(y, np.ndarray) and isinstance(X, np.ndarray), "Expect numpy.ndarray input"
        assert X.shape[0] == y.shape[0], "Samples sizes don't match"
        # Check for binary y
        unique_values = np.unique(y)
        assert len(unique_values) >= 2, "y must contain at least two unique values for classification"
        # Float 
        X, y = X.astype(float), y.astype(float)
        if len(y.shape)  == 1: 
            y = y.reshape((y.shape[0],1))
        if len(X.shape) == 1: 
            X = X.reshape((X.shape[0],1))
        if y.shape[1] == 1: 
            if not np.array_equal(np.unique(y[:,0]), np.array([0.,1.])): 
                y = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y)
                if y.shape[1] == 2:
                    y = y[:,0].reshape((y.shape[0],1))
        if y.shape[1] == 1:
            self._type = "binary"
        else:
            self._type = "multiclass"
        return X, y
    
class GiniAFD(Model):
    
    def __init__(self, gini_param=2, recompute=True):
        super().__init__()
        self.gini_param = gini_param
        self.recompute = recompute
        self.nb_components = None
    def _rank(self, x):
        if x.ndim == 1:
            rank = (x.shape[0] + 1 - ss.rankdata(x, method='average'))
            return rank
        else:
            n, k = x.shape
            rank = np.zeros_like(x)
            for i in range(k):
                rank[:,i] = (n + 1 - ss.rankdata(x[:,i], method='average'))
            return rank
        
    def fit(self, X, y, nb_components = None):
        if nb_components is not None:
            self.nb_components = nb_components  
            assert self.nb_components <= X.shape[1], "nb_components must be less than or equal to the number of features in X"
        self.nb_components = nb_components
        X, y = self._preprocess(X, y)
        self.group_size = self._group_size(y)
        self._fit(X, y)
        return self.w
    
    def _fit(self, X, y, recompute=True):
        n = X.shape[0]
        self.y = y
        # X
        self.X = X
        self.X_mean = X.mean(0)
        self.X_centered = X - self.X_mean
        self.X_group_mean = self._group_mean(X,y,target="X")
        self.X_group_centered = self._group_centered(X,y,self.X_group_mean)
        # Rank matrix
        self.rank = self._rank(X)**(self.gini_param-1)
        self.rank_mean = self.rank.mean(0)
        self.rank_centered = self.rank - self.rank_mean
        #self.rank_group_mean = self._group_mean(X,y)
        self.rank_group_mean = self._group_mean(self.rank,y)
        self.rank_group_centered = self._group_centered(self.rank,y,self.rank_group_mean)
        # Gini cov
        self.gcov = -2*self.gini_param*self.X.T @ self.rank/(n*(n-1))
        self.gcov_centered = -2*self.gini_param*self.X_centered.T @ self.rank_centered/(n*(n-1))
        self.gcov_group_centered = -2*self.gini_param*self.X_group_centered.T @ self.rank_group_centered/(n*(n-1))
        self.gcov_group =  self._group_cov(self.X_group_centered,self.rank_group_centered,y)
        # Gini cor
        self.gcor = self.gcov / np.diag(self.gcov)
        self.gcor_centered = self.gcov_centered / np.diag(self.gcov_centered)
        self.gcor_group_centered = self.gcov_group_centered / np.diag(self.gcov_group_centered)
        # Z
        self.Z = self.X_centered/np.diag(self.gcov_centered)
        self.Z_group_mean = self._group_mean(self.Z,y,target="Z")
        self.Z_group_centered = self._group_centered(self.Z,y,self.Z_group_mean)
        # Group matrix
        self.g_within = self.Z_group_centered.T.dot(self.rank_group_centered)
        self.g_within *= -2*self.gini_param/n/(n-1)
        self.g_between = -2*self.gini_param/n/(n-1)*self.Z.T.dot(self.rank_centered) - self.g_within
        # GW
        g_wb = pinv(self.g_within + self.g_within.T).dot(self.g_between + self.g_between.T)
        # Sorted eigenvalues
        valp, vecp = np.linalg.eig(g_wb)
        idx = valp.argsort()[::-1]
        valp = valp[idx]
        vecp = vecp[:,idx]
        A = np.zeros((g_wb.shape[1],3)) 
        A[:,0] = np.real(valp.T)
        A[:,2] = (np.cumsum(A[:,0], axis=0) / np.sum(A[:,0]))*100
        A[:,1] = (A[:,0] / A[:,0].sum())*100
        self.df_eigen = pd.DataFrame(np.round(A,4), columns=['Eigenvalues', '%', 'Cumsum %'])
        self.df_eigen_not_full = pd.DataFrame(np.round(A[0:self.y.shape[1]-1,:],4), columns=['Eigenvalues', '%', 'Cumsum %'])
        
        # Projection
        if self.nb_components:
            self.w = vecp[:, :self.nb_components]
        else:
            self.w = vecp[:, :y.shape[1]]
        self.W = vecp[:, max(1, y.shape[1]-1)]
        if max(1, y.shape[1]-1) == 1: self.W = self.W[:,np.newaxis]

        # f: discriminant axes
        self.f = np.real(self.Z.dot(self.w))
        self.discriminant_axes = self.f
        self.f_group_mean = self._group_mean(self.f,y,target="F")
        self.f_group_centered = self._group_centered(self.f,y,self.f_group_mean)
        
        # V_w
        self.V_w = 0
        for i in range(len(self.gcov_group)):
            self.V_w += 1/X.shape[0] * self.group_size[i] * self.gcov_group[i]
        return self.Z.dot(vecp)
    
    # g_0 in the paper with gcov instead => equivalent to g_6 below
    def euclidean(self, X): 
        outputs = []
        pinv_cov = pinv(self.V_w)
        for i in range(len(self.gcov_group)):
            X_ = X - self.X_group_mean[i]
            outputs.append(np.diag(X_.dot(pinv_cov).dot(X_.T)))
        return np.array(outputs).T
    
    # g_1 in the paper
    def euclidean_gini(self, X, rank):
        outputs = []
        for i in range(len(self.gcov_group)):
            X_ = X - self.X_group_mean[i]
            R_ = rank - self.rank_group_mean[i]
            outputs.append(np.diag(X_.dot(pinv(self.gcov_centered)) @ R_.T))
        return np.array(outputs).T
    
    # g_3 in the paper  
    def geometric_gini(self, F, group_means):
        scores = np.zeros((F.shape[0], len(group_means)))
        for i, (label, mean_vector) in enumerate(group_means):
            scores[:, i] = np.sum(np.abs(F - mean_vector), axis=1)
        lowest_scores_idx = np.argmin(scores, axis=1)
        groups_with_lowest_scores = [group_means[i][0] for i in lowest_scores_idx]
        groups_with_lowest_scores_array = np.array(groups_with_lowest_scores)    
        return groups_with_lowest_scores_array.T       
                    
    # g_6 in the paper : replace cov by gcov in the homosc. LDA 
    def homo_gini(self, X):
        outputs = []
        V_w = 0
        for i in range(len(self.gcov_group)):
            V_w += 1/X.shape[0] * self.group_size[i] * self.gcov_group[i]
        for i in range(len(self.gcov_group)):
            X_ = X - self.X_group_mean[i]
            output = -2*np.log(self.group_size[i]/self.group_size.sum()) 
            outputs.append(np.diag(output + X_.dot(pinv(V_w)).dot(X_.T)))
        return np.array(outputs).T
    
    # g_7 in the paper : replace cov_k by gcov_k (heterosc. Gini)
    def hetero_gini(self, X):
        outputs = []
        for i in range(len(self.gcov_group)):
            X_centered = X - self.X_group_mean[i]
            output = -2*np.log(self.group_size[i]/self.group_size.sum()) 
            output += np.log(np.abs(np.linalg.det(self.gcov_group[i]))) 
            outputs.append(np.diag(output + X_centered @ pinv(self.gcov_group[i]) @ X_centered.T))
        return np.array(outputs).T
    
    def predict(self, X, y, method = True):
        '''
        ------------------
        Function predict:
        ------------------
            - method = 'euclidean_gini'
            - method = 'geometric_gini'
            - method = 'homo_gini' 
            - method = 'hetero_gini' 
        '''
        self.method = method
        assert self.w.any(), "Fit before predict"
        Z = (X - self.X_mean)/np.diag(self.gcov_centered)
        f = Z.dot(self.w)
        X_cat = np.concatenate((X, self.X), axis=0)
        ranks = (self._rank(X_cat)/X_cat.shape[0]*self.X.shape[0])**(self.gini_param-1)
        ranks = ranks[:X.shape[0]]
        Xf_cat = ((X_cat - self.X_mean)/np.diag(self.gcov_centered)).dot(self.w)
        ranksf = (self._rank(Xf_cat)/Xf_cat.shape[0]*self.X.shape[0])**(self.gini_param-1)
        ranksf = ranksf[:X.shape[0]]
        
        # Identify group means of f (discriminant axes) with labels of y_test: see "geometric_gini"
        unique_labels = np.unique(y)
        group_means = []  
        for label in unique_labels:
            indices = np.where(y == label)[0]        
            mean = np.mean(f[indices], axis=0) 
            group_means.append((label, mean))  
            
        # Outliers
        self.nb_outliers = self.outliers(X)
        
        # Methods
        if method=="gini_estimator":
            output = self.euclidean(X)
            return np.argmin(output, -1)
        
        elif method=="euclidean_gini":
            output = self.euclidean_gini(X, ranks)
            return np.argmin(output, -1)

        elif method=="geometric_gini":
            output = self.geometric_gini(f, group_means)
            return output
        
        elif method=="homo_gini":
            output = self.homo_gini(X)
            return np.argmin(output, -1)

        elif method=="hetero_gini":
            output = self.hetero_gini(X)
            return np.argmin(output, -1)

    def _group_size(self, y):
        if self._type == 'binary':
            y = np.concatenate((y,1-y), axis=1)
        return np.expand_dims(y.sum(axis=0), axis=1)
    
    # compure rank conditional to group k
    def _rank_group(self, base_X, X, y):
        if y.shape[1] == 1: 
            y = np.concatenate((y,1-y), axis=1)
        outputs = []
        for i in range(y.shape[1]):
            X_group = base_X*y[:,i][:,np.newaxis]
            X_cat = np.concatenate((X_group, X), axis=0).dot(self.W)
            v = (self._rank(X_cat)/X_cat.shape[0]*self.group_size[i])**(self.gini_param-1)
            v = v[:X_group.shape[0]]
            v = v - self.rank_group_mean[i]**(self.gini_param-1)
            outputs.append(v[X_group.shape[0]:])
        return outputs

    def _group_mean(self, X, y, rank=None, target="rank"):
        # Compute mean for each group
        if y.shape[1] == 1: 
            y = np.concatenate((y,1-y), axis=1)
        group_means = []
        for i in range(y.shape[1]):
            if rank is not None:
                current_mean = rank * y[:,i][:,np.newaxis]
                current_mean = current_mean[~np.all(current_mean==0, axis=1)]
            else:
                current_X = X * y[:,i][:,np.newaxis]
                current_X = current_X[~np.all(current_X==0, axis=1)]
                if target == "rank":
                    current_mean = self._rank(current_X)**(self.gini_param-1)
                else:
                    current_mean = current_X
            group_means.append(current_mean.mean(0))
        return group_means
    
    def _group_centered(self, X, y, groups):
        if y.shape[1] == 1: 
            y = np.concatenate((y,1-y), axis=1)
        output = 0
        for i in range(y.shape[1]):
            output += (X-groups[i])*y[:,i][:,np.newaxis]
        return output
    
    def _group_cov(self, X1, X2, y):
        if y.shape[1] == 1: 
            y = np.concatenate((y,1-y), axis=1)
        n = X1.shape[0]
        all_cov = []
        for i in range(y.shape[1]):
            current_X1 = X1 * y[:,i][:,np.newaxis]
            current_X1 = current_X1[~np.all(current_X1==0, axis=1)]
            current_X2 = X2 * y[:,i][:,np.newaxis]
            current_X2 = current_X2[~np.all(current_X2==0, axis=1)]
            if current_X1.size > 0 and current_X2.size > 0:
                all_cov.append(current_X1.T.dot(current_X2)/n)
            else:
                all_cov.append(0)
        return all_cov
    
    def outliers(self, x):
        n,k = x.shape
        outliers_variables = []
        for i in range (1,k):
            outliers_variables.append(grubbs.max_test_indices(x[:,i], alpha = 0.1))
        if len(outliers_variables) > 0:
            self.number_outliers = len(outliers_variables)
            return self.number_outliers
    
    def grid_search(self, x_train, y_train, x_test, y_test, nu_values, classif_methods, nb_components = None):
        accuracy_matrix = [] 
        y_pred_matrix = [] 
        for nu in nu_values:
            accuracies_for_current_nu = []
            y_pred = [] 
            for method in classif_methods:
                self.gini_param = nu
                if nb_components is not None:
                    self.fit(x_train, y_train, nb_components = nb_components)
                else:
                    self.fit(x_train, y_train)
                pred = self.predict(x_test, y_test, method=method)
                y_pred.append(pred)
                acc = accuracy_score(y_test.flatten(), pred)
                accuracies_for_current_nu.append(acc)
            accuracy_matrix.append(accuracies_for_current_nu)
            y_pred_matrix.append(y_pred)
        accuracy_matrix = np.array(accuracy_matrix)
        y_pred_matrix = np.array(y_pred_matrix)
        
        # Iterate over the accuracy matrix and nu_values simultaneously
        best_nu = None
        max_accuracy = 0
        for nu, accuracies, yp in zip(nu_values, accuracy_matrix, y_pred_matrix):
            max_accuracy_index = np.argmax(accuracies)
            current_max_accuracy = np.max(accuracies)
            if current_max_accuracy > max_accuracy:
                y_pred = yp[max_accuracy_index]
                max_accuracy = current_max_accuracy
                best_nu = nu
                best_method = classif_methods[max_accuracy_index]
        
        data = {
            '  Best nu:': best_nu,  
            '    Max accuracy:': max_accuracy,
            '            Method:': best_method,
        }
        df = pd.DataFrame([data])
        separator_line = '=' * 55
        print(separator_line)
        print(f"                  Gridsearch: Best Model")
        print(separator_line)
        print(df)
        print(" ")
        separator_line_2 = '=' * 55
        print(separator_line_2)
        print(f"                        Performance")
        report = classification_report(y_test, y_pred)
        print(separator_line_2)
        print(report)

    
    def grid_search_kfold(self, x, y, nu_values, classif_methods, nb_components=None, n_splits=5):
        accuracy_results = []  
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for nu in nu_values:
            for method in classif_methods:
                accuracies = []
                for train_index, test_index in kf.split(x):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    self.gini_param = nu
                    if nb_components is not None:
                        self.fit(x_train, y_train, nb_components=nb_components)
                    else:
                        self.fit(x_train, y_train)
                    pred = self.predict(x_test, y_test, method=method)
                    acc = accuracy_score(y_test.flatten(), pred)
                    accuracies.append(acc)
                # Store the mean accuracy
                accuracy_results.append({
                    "nu": nu,
                    "method": method,
                    "mean_accuracy": np.mean(accuracies)
                })
        # Best nu
        results_df = pd.DataFrame(accuracy_results)
        best_result = results_df.loc[results_df["mean_accuracy"].idxmax()]
        best_nu = best_result["nu"]
        best_method = best_result["method"]
        max_accuracy = best_result["mean_accuracy"]
        
        # Run KFold with the best nu and method 
        y_true_all, y_pred_all = [], []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.gini_param = best_nu
            if nb_components is not None:
                self.fit(x_train, y_train, nb_components=nb_components)
            else:
                self.fit(x_train, y_train)
            
            pred = self.predict(x_test, y_test, method=best_method)
            y_true_all.extend(y_test.flatten())
            y_pred_all.extend(pred)
        
        # Results
        report = classification_report(y_true_all, y_pred_all)
        separator_line = '=' * 55
        print(separator_line)
        print("                  Gridsearch: Best Model")
        print(separator_line)
        print(f"Best nu: {np.round(best_nu,2)}")
        print(f"Max accuracy: {np.round(max_accuracy,2)}")
        print(f"Method: {best_method}")
        print(separator_line)
        print("                        Performance")
        print(separator_line)
        print(report)
        print(separator_line)
    
    
    def summary(self):
        nb_outliers = self.outliers(self.X)
        table_data = [
            ["Target Variable:", "y", "No. of features", self.X.shape[1]],
            ["No. of classes", self.y.shape[1],"Sample size", self.X.shape[0]],
            ["No. Outliers in X:", nb_outliers, "Gini parameter (nu)", self.gini_param],
            ["Date:", datetime.now().strftime("%a, %d %b %Y"), "Time:", datetime.now().strftime("%H:%M:%S")]
            ]
        header = ["Generalized Gini Discriminant Analyses"]
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
        
        # Table of eigenvalues
        print("")
        separator = '=' * 40
        print(separator)
        print("        Table of Eigenvalues")
        print(separator)
        if self.nb_components:
            print(self.df_eigen_not_full)
        else:
            print(self.df_eigen)
        print(separator)
        print("")
        
        # Table of correlations
        results = []
        if self.nb_components:
            comp = self.nb_components
        elif self.y.shape[1] == 1:
            comp = 1
        else:
            comp = 3
        dataframes = []  
        for i in range(comp):
            correlations = []
            p_values = []
            for j in range(self.X.shape[1]):
                r, p = scipy.stats.pearsonr(self.f[:,i], self.X[:,j])
                correlations.append(round(r, 4))
                p_values.append(round(p, 4))
            
            # Create a DataFrame for this component
            df = pd.DataFrame({
                "Correlation": correlations,
                "p_value": p_values
            }, index=[f"X{j+1}" for j in range(self.X.shape[1])])
            
            separator_line = '-' * 35
            #print(separator_line)
            print(f"Stats for discriminant axis {i+1}:")
            print(separator_line)
            print(df)
            print(separator_line)
            dataframes.append(df)
