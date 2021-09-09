import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from seaborn import matrix
warnings.filterwarnings('ignore')
matrix_1 = (np.random.random((10,10)) - 0.5) * 2
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(matrix_1,cmap=plt.cm.RdYlBu_r,
                    vmin = -1., vmax = 1., annot = True)
plt.show()
nrow = 1000
matrix_2 = (np.random.random((1000,3))-0.5)*2
indice  = np.random.choice([0,1,2],size=nrow)
plot_data = pd.DataFrame(matrix_2, indice).reset_index()
print(plot_data.head(5))
print(plot_data.columns)#['index', 0, 1, 2]
grid = sns.PairGrid(data = plot_data,size=3,diag_sharey=False,
                    hue = 'index',
                    vars = [x for x in list(plot_data.columns) if x != 'index'])

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
plt.show()

#matrix_3 = np.random.normal(-1.0,1.0,1000*3).reshape((1000,3))
matrix_3 = np.random.randn(nrow * 3).reshape((nrow, 3))
indice  = np.random.choice([0,1,2],size=nrow)
plot_data_2 = pd.DataFrame(matrix_3, indice).reset_index()
grid_2 = sns.PairGrid(data = plot_data,size=3,diag_sharey=False,
                    hue = 'index',
                    vars = [x for x in list(plot_data_2.columns) if x != 'index'])

grid_2.map_upper(plt.scatter, alpha = 0.2)
grid_2.map_diag(sns.kdeplot)
grid_2.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
plt.show()