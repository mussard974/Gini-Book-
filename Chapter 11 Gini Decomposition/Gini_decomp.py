"""
Gini Decomposition
--------
This module implements the (α, β)-Gini decomposition of inequality, a framework 
for breaking down inequality into within-group, between-group, and transvariation 
components. It follows the methodology developed in:

    - Mornet P., Zoli C., Mussard S., Sadefo-Kamdem J., Seyte F., Terraza M. (2013), 
      "The (α, β)-multi-level α-Gini decomposition with an illustration to income 
      inequality in France in 2005." Economic Modelling, vol. 35 (C), pp. 944–963.
    - Mussard, S. and Mornet, P. (2019), 
      "A Note on α-Gini Measures." Review of Income and Wealth, 65: 675–682.

Features:
    - Compute intra-group and inter-group Gini indices
    - Decompose inequality into:
        * Within-group Gini (Gw)
        * Between-group Gini (Ggb / Gnb)
        * Transvariation component (Gt)
    - Support for α and β parameterization of Gini measures
    - Compute correlation ratios (ANOGI interpretation)
    - Generate summary tables with PrettyTable
    - Export weighted Gini indices and between-group distance matrices
    - Provide formatted outputs for inspection in console or Jupyter notebooks

Dependencies:
    - pandas
    - torch
    - prettytable

Class:
    GiniDecomposition:
        Provides methods for fitting the decomposition to a dataset, 
        computing group-level indices, and producing formatted summaries.
"""

import pandas as pd
import torch
from prettytable import PrettyTable
pd.options.display.float_format = '{:.4f}'.format

class GiniDecomposition(object):
    
    '''
    Gini (alpha,beta) decomposition
    
    cite : Mornet P., Zoli C., Mussard S., Sadefo-Kamdem J., Seyte F., Terraza M. (2013), The (α, β)-multi-level α-Gini decomposition with an illustration to income inequality in France in 2005. Economic Modelling, vol. 35 (C), pp. 944-963.
    cite : Mussard, S. and Mornet, P. (2019), A Note on α-Gini Measures. Review of Income and Wealth, 65: 675-682. https://doi.org/10.1111/roiw.12373
    ----------
    Functions:
    ----------
    fit :
        => name of the dataframe + name of the value column to compute the Gini index + name of the group column
        => example : fit(df, 'Values', 'Group') 
    '''
    
    def __init__(self, alpha, beta = None, method = None):
        self.alpha = alpha
        self.beta = beta
        self.method = method
        
    def gini_inter(self, x, y):
        combinations = torch.cartesian_prod(x, y)
        if combinations[:, 0].mean() > combinations[:, 1].mean():
            differences = combinations[:, 0] - combinations[:, 1]
        else:
            differences = combinations[:, 1] - combinations[:, 0]
        positive_diff = differences[differences > 0].sum()
        negative_diff = differences[differences < 0].sum()
        Gini = (torch.abs(differences)**self.alpha).sum() / ((torch.mean(x)**self.alpha + torch.mean(y)**self.alpha)*(len(x)*len(y)))
        Distance = (positive_diff**self.alpha - (-1*negative_diff)**self.alpha) / (positive_diff**self.alpha + (-1*negative_diff)**self.alpha)
        if self.beta:
             Distance = (positive_diff**self.beta - (-1*negative_diff)**self.beta) / (positive_diff**self.beta + (-1*negative_diff)**self.beta)
        return Gini, Distance

    def gini_intra(self, x):
        pairs = torch.combinations(x)
        distances = torch.abs(pairs[:,0]-pairs[:,1])**self.alpha
        Gini = torch.sum(2*distances) / (torch.mean(x)**self.alpha*2*len(x)**2)
        return Gini
    
    def tensor_from_pandas(self, dataframe, values_column, group_column):
        dataframe[group_column] = dataframe[group_column].astype(str)
        groups = dataframe[group_column].unique()
        tensors_list = []
        means = []
        sizes = []
        total_mean = []
        for group in groups:
            group_data = dataframe[dataframe[group_column] == group][values_column]
            tensor = torch.tensor(group_data.values, dtype=torch.float64)  
            tensors_list.append(tensor)
            # Size share and sum share per group
            group_sum_ratio = len(group_data)*(tensor.mean().item()**self.alpha) / (len(dataframe)*dataframe[values_column].mean()**self.alpha)
            group_size_ratio = len(group_data) / len(dataframe)
            sizes.append(group_size_ratio)
            means.append(group_sum_ratio)
            total_mean.append(len(group_data) / len(dataframe) * tensor.mean().item())
        means_tensor = torch.tensor(means, dtype=torch.float64)
        size_tensor = torch.tensor(sizes, dtype=torch.float64)
        self.total_mean = sum(total_mean)
        return tensors_list, means_tensor, size_tensor
    
    def fit(self, dataframe, values_column, group_column):
        self.dataframe = dataframe
        self.group_column = group_column
        tensors_list = self.tensor_from_pandas(self.dataframe, values_column, self.group_column)[0]
        self.num_tensors = len(tensors_list)
        self.means_tensor = self.tensor_from_pandas(self.dataframe, values_column, self.group_column)[1]
        self.means_tensor = self.means_tensor.clone().detach()
        self.size_tensor = self.tensor_from_pandas(self.dataframe, values_column, self.group_column)[2]
        self.size_tensor = self.size_tensor.clone().detach()
        self.tensor_Gjj = [self.gini_intra(tensor) for tensor in tensors_list]
        self.tensor_Gjj = torch.stack(self.tensor_Gjj)
        self.matrix_Gij = torch.zeros((self.num_tensors, self.num_tensors), dtype=float)
        self.matrix_Dij = torch.zeros((self.num_tensors, self.num_tensors), dtype=float)
        for i in range(self.num_tensors - 1):
            for j in range(i + 1, self.num_tensors):
                Gij = self.gini_inter(tensors_list[i], tensors_list[j])[0]
                Dij = self.gini_inter(tensors_list[i], tensors_list[j])[1]
                self.matrix_Gij[i,j] = Gij
                self.matrix_Gij[j,i] = Gij 
                self.matrix_Dij[i,j] = Dij
                self.matrix_Dij[j,i] = Dij
        self.Gini_between = self.means_tensor @ self.matrix_Gij @ self.size_tensor
        self.Gini_net_between = self.means_tensor @ (self.matrix_Gij * self.matrix_Dij) @ self.size_tensor
        self.Gini_transvariation = self.means_tensor @ (self.matrix_Gij * (1-self.matrix_Dij)) @ self.size_tensor
        self.Gini_within = torch.sum(self.means_tensor * self.tensor_Gjj * self.size_tensor)
        self.Gini_total = self.Gini_between + self.Gini_within
        if self.method == "absolute":
            self.Gini_within = self.Gini_within * 2 * self.total_mean**self.alpha
            self.Gini_net_between = self.Gini_net_between * 2 * self.total_mean**self.alpha
            self.Gini_between = self.Gini_between * 2 * self.total_mean**self.alpha
            self.Gini_transvariation = self.Gini_transvariation * 2 * self.total_mean**self.alpha
            self.Gini_total = self.Gini_total * 2 * self.total_mean**self.alpha
    
    def correlation_ratio(self):
        return self.Gini_between.item() / self.Gini_total.item()
            
    def summary(self):
        
        # Table with Gw and Ggb        
        Gw = f"{self.Gini_within.item():.4f}"
        Ggb = f"{self.Gini_between.item():.4f}"
        G = self.Gini_within + self.Gini_between
        table = PrettyTable()
        table.field_names = ["Gini decomposition: 2 components", "  Index  "]
        table.add_row(["Gini within groups (Gw)", Gw])
        table.add_row(["    Gini between groups (Ggb)    ", Ggb])
        table.add_row(["Gini total (Gw + Ggb)", f"{G.item():.4f}"])
        print(table)

        # Table with Gw, Gnb and Gt
        Gnb = f"{self.Gini_net_between.item():.4f}"
        Gt = f"{self.Gini_transvariation.item():.4f}"
        G = self.Gini_within + self.Gini_net_between + self.Gini_transvariation
        table = PrettyTable()
        table.field_names = ["Gini decomposition: 3 components", "  Index  "]
        table.add_row(["Gini within groups (Gw)", Gw])
        table.add_row(["  Gini net between groups (Gnb)  ", Gnb])
        table.add_row(["Gini transvariation (Gt)", Gt])
        table.add_row(["Gini total (Gw + Gnb + Gt)", f"{G.item():.4f}"])
        print(table)
        
        if self.method:
            self.correlation_ratio = self.Gini_net_between.item() / self.Gini_total.item()
            table = PrettyTable()
            table.field_names = ["ANOGI", "  Index  "]
            table.add_row(["GMD net between groups (GMDnb)", f"{self.Gini_net_between.item():.4f}"])
            table.add_row(["GMD total", f"{self.Gini_total.item():.4f}"])
            table.add_row(["Correlation ratio (GMDnb / GMD)", f"{self.correlation_ratio:.4f}"])
            print(table)
            
        if self.method is None:
            # Table with Gjh
            group_names = [str(name) for name in self.dataframe[self.group_column].unique()]
            matrix_data = [[float(tensor) for tensor in row] for row in self.matrix_Gij]
            df = pd.DataFrame(matrix_data, index=group_names, columns=group_names)
            styles = [{'selector': 'th.col_heading', 'props': 'text-align: center;'}]
            try:
                styled_df = df.style.set_table_styles(styles)
                styled_df = styled_df.format('{:.4f}')
                print(f"\n{'Matrix of between-group Gini indices G_gh:'}")
                display(styled_df)
            except Exception:
                print(f"\n{'Matrix of between-group Gini indices G_gh:'}")
                display(df)
            
            # Table with Gw
            Gjj_weighted = self.means_tensor * self.tensor_Gjj * self.size_tensor
            Gjj_weighted_data = Gjj_weighted.squeeze().tolist()
            df = pd.DataFrame({group: [value] for group, value in zip(group_names, Gjj_weighted_data)})
            try:
                styled_df = df.style.set_table_styles(styles)
                df.index = ["Weighted Gini within"] * len(df)
                df.reset_index(drop=True, inplace=True)
                print(f"{'Weighted Gini indices p^g*s^g*G_gg:'}")
                styled_df = styled_df.format('{:.4f}')
                display(styled_df)
            except Exception:
                print(f"{'Weighted Gini indices p^g*s^g*G_gg:'}")
                display(df)
                
            # Table with Djh
            group_names = [str(name) for name in self.dataframe[self.group_column].unique()]
            matrix_data = [[float(tensor) for tensor in row] for row in self.matrix_Dij]
            df = pd.DataFrame(matrix_data, index=group_names, columns=group_names)
            styles = [{'selector': 'th.col_heading', 'props': 'text-align: center;'}]
            try:
                styled_df = df.style.set_table_styles(styles)
                styled_df = styled_df.format('{:.4f}')
                print(f"{'Matrix of Distances D_gh:'}")
                display(styled_df)
            except Exception:
                print(f"{'Matrix of Distances D_gh:'}")
                display(df)
            
            # Table with summary (sj, pj, Gjj)
            header = f"{'Summary Table':^45}\n{'=' * 51}\n{'Groups':<15}{'Means':<15}{'Size':<15}{'Gini':<20}\n{'-' * 51}"
            rows = ""
            for i in range(self.num_tensors):
                gini_formatted = f"{self.tensor_Gjj[i].item():.4f}"  
                rows += f"{str(self.dataframe[self.group_column].unique()[i]):<15}{self.means_tensor[i].item():<15.4f}{self.size_tensor[i].item():<15.4f}{gini_formatted}\n"
            print(f"{'=' * 51}\n{header}\n{rows}\n{'=' * 51}")
                       
