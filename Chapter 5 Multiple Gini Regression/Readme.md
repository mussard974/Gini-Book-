---
runme:
  id: 01HSWZVF1HY3JSBYH7JJY8DEP0
  version: v3
---

# Gini Multiple Regression

---

### Robust regression with the Gini metric


---

*class* `GiniRegression`

`fit(y, x, add_constant = False, parametric_estimator = False, cov = False, iter = 1000)`

In this package, we find in the ***fit*** function: 

* Use intercept in the regression :  
    - **add_constant :** ***bool, default = True*** 

* The non-parametric Gini estimator with Jackknife tests 
    - **iter :** ***float, default is the size of the sample*** 
    
        Fix the number of iterations for the std.dev. Jackknife estimator (non parametric method only)
         
* The parametric Gini estimator (bool) : default False in this case the non parametric estimator is used.  
    - **parametric_estimator :** ***bool, default = True*** 
    
        The solver minimizes the co-Gini of the errors with BFGS (Broyden–Fletcher–Goldfarb–Shanno) gradient descent of scipy
        
* The asymptotic covariance of the estimates        
    - **cov :** ***{'asympt', 'instrument', 'bandwidth iid'} default is Jackknife variance of the non parametric beta Gini estimators***
    
    This measures the variance-covariance matrix for parametric estimation only. 'asympt' for asymptotic estimation like OLS : $\sigma_e^2 (X'X)^{-1}$. 'instrument' for asymptotic estimation like in OLS with $rank(X) = Z$ as instrument:
    $$ 
    cov = \sigma_e^2 (Z'X)^{-1} @ (X'X) @ (X'Z)^{-1}
    $$
    'bandwidth iid' see statsmodels fo quantile regressions : https://www.statsmodels.org/dev/_modules/statsmodels/regression/quantile_regression.html
        

### Install

```python {"id":"01HSX1JAZVREPPXVJG5ZPJR5VP"}
!pip install outliers
!pip install outlier_utils
!pip install iteration_utilities
!pip install openpyxl
!pip install mlxtend
!pip install torch
```

### Import Multiple Gini Regression

```python {"id":"01HSX1KRCAPE9XEBJCPTY00M39"}
from GiniRegression import GiniRegression
```

###  Usage: an example with `fetch_california_housing`

<div align="center"> 
<a href="http://scikit-learn.org/stable/#"><img src="http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" style="max-width: 180px; display: inline" alt="Scikit-Learn"/></a>
</div>

```python {"id":"01HSWZVF1HY3JSBYH7J98555TJ"}
from sklearn.datasets import fetch_california_housing
data_california = fetch_california_housing()
X = data_california.data  
y = data_california.target 
```

### Non-parametric Gini regression

```python {"id":"01HSX23FZ1XYKC68YDH1YSPY1G"}
# Non-parametric Gini regression 
model = GiniRegression()
model.fit(y, X)
model.summary()
```

### Non-parametric Gini regression with iterations for Jackknife

```python {"id":"01HSX24F02H63RHGRZ58KM1Y0D"}
# Non-parametric Gini regression 
model = GiniRegression()
model.fit(y, X, iter = 1000)
model.summary()
```

### Parametric Gini regression with instruments

```python {"id":"01HSWZVF1HY3JSBYH7JBXMDQ6V"}
# Instrument is the rank matrix of X for covariance estimates of the parameters
model = GiniRegression()
model.fit(y, X, parametric_estimator=True, cov='instrument') 
model.summary()
```

### Parametric Gini regression with Jackknife

```python {"id":"01HSWZVF1HY3JSBYH7JCFXNEX3"}
# Gini parametric with Jackknife covariance for coefficient estimates
model = GiniRegression()
model.fit(y, X, parametric_estimator=True, cov='Jackknife') 
model.summary()
```


