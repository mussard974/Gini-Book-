# Gini-PCA
A. Charpentier, S. Mussard & T. Ouraga (2020)
***

### European Journal of Operational Research
[open access to the paper](https://www.sciencedirect.com/science/article/pii/S0377221721000886)


### Gini PCA is a robust L1-norm PCA based on the generalized Gini index

In this package, we find:

  * Grid search for detecting and minimizing the influence of outliers 
  * Absolute contributions of the observations 
  * Relative contributions of the observations (equivalent to cos²)
  * Feature importance of each variable (U-statistics test)
  * Outlier detection using Grubbs test 
  * Example on cars data


### Install outlier_utils and iteration-utilities
* torch >= 1.9.0
```python
!pip install outlier_utils
!pip install iteration-utilities
```

### Import Gini PCA


```python
from Gini_PCA import GiniPca
```

### Import data and plot utilities: example on cars data


```python
from Gini_PCA import GiniPca
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
cars = pd.read_excel("cars.xlsx", index_col=0)
type(cars)
x = torch.DoubleTensor(cars.values)
```

### Run the model by setting your own Gini parameter >=0.1 and != 1

```python
gini_param = 2
model = GiniPca(gini_param)
```


### Ranks matrix

```python
model.ranks(x)
```


### Gini mean difference (GMD) matrix

```python
model.gmd(x)
```


### Gini correlation matrix

```python
z = model.scale_gini(x)
print('Gini correlation:', '\n', model.gmd(z))
```

Gini correlation: 
 tensor([[1.0000, 0.9643, 0.9025, 0.7347, 0.8840, 0.7317],
        [0.9554, 1.0000, 0.9647, 0.5606, 0.8245, 0.6209],
        [0.8472, 0.9482, 1.0000, 0.4510, 0.6802, 0.5822],
        [0.8111, 0.6868, 0.5071, 1.0000, 0.8299, 0.8179],
        [0.8366, 0.7860, 0.6570, 0.5681, 1.0000, 0.7433],
        [0.8276, 0.7268, 0.6787, 0.8520, 0.8511, 1.0000]], dtype=torch.float64)





### Otherwise find the optimal Gini parameter: Grid search


```python
parameter = model.optimal_gini_param(x)
print(parameter)
```

    0.1
    

### Project the data x onto the new subspace


```python
scores = model.project(x)
```

### 3D plot


```python
y = iris.target
model.plot3D(x, y) 
```


![png](output_14_0.png)


### Absolute contributions in %


```python
model.act(x)*100
```




    array([[ 4.14204577e-01,  3.36374074e-01,  1.18452709e+00,
            -6.10196220e-02],
           [ 6.03226665e-01, -9.35206328e-02, -1.40846480e-01,
             6.88762710e-03],
           [ 1.73121668e+00, -8.75352630e-03, -8.74836712e-01,
            -8.07322629e-01],
           [ 1.09951951e+00,  3.01913547e-02, -1.76259247e+00,
             2.33765060e+00],
             
             ...
             
           [ 5.14220524e-01,  3.79632062e-01,  4.06749751e-01,
             1.10654297e-02],
           [-1.90812939e-01,  8.13731181e-01, -1.25608092e-01,
             1.31341783e-01], 1.19623781e-01]])



### Relative contributions 


```python
model.rct(x)
```




    array([[9.83548621e-03, 6.62780300e-03, 7.31376244e-03, 3.55074089e-03],
           [9.79396153e-03, 3.63358371e-03, 8.43032568e-03, 3.07092805e-04],
           [1.06682515e-02, 4.60316819e-04, 3.77033540e-03, 2.55039508e-03],
           [1.04770642e-02, 2.80338240e-03, 2.56572182e-03, 7.13775902e-03],
           [1.01898345e-02, 8.19383690e-03, 4.86520485e-03, 5.80896914e-03],
           [8.30159091e-03, 1.52459327e-02, 4.46401173e-03, 2.61005226e-03],
           [1.06020351e-02, 3.00105569e-03, 1.50962919e-03, 3.81757834e-03],
           [9.80989222e-03, 4.32593652e-03, 6.34627377e-03, 5.86495062e-03],
           
           ...
           
           [1.09226562e-02, 7.28351809e-03, 1.34211922e-03, 4.94671582e-03],
           [1.00976884e-02, 1.78461923e-03, 9.06928902e-03, 8.06161896e-03],
           [9.14272028e-03, 1.14552554e-02, 1.01268257e-02, 4.59214222e-03],
           [4.59906381e-03, 1.31378979e-03, 1.22116519e-02, 2.95281106e-03]])



### Feature importance in Gini PCA (factors loadings)

* U-statistics > 2.57: significance of 1% 
* U-statistics > 1.96: significance of 5% 
* U-statistics > 1.65: significance of 10% 


```python
model.u_stat(x)
```




    array([[-40.30577176,   0.94614065, -48.12517983, -45.22470418],
           [ -3.12762227, -23.76476321,   1.35866338,   1.26459391]])



### Feature importance in standard PCA (factors loadings)

* U-statistics > 2.57: significance of 1% 
* U-statistics > 1.96: significance of 5% 
* U-statistics > 1.65: significance of 10% 


```python
model.u_stat_pca(x)
```




    array([[11.78270546, -5.86022403, 15.57293717, 15.58145032],
           [-4.06529552, -7.43327168, -0.27455139, -0.79030779]])




### Grubbs test


```python
model.optimal_gini_param(x)
```

[Stéphane Mussard CV_HAL](https://cv.archives-ouvertes.fr/stephane-mussard)

