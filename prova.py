import sys
import os

# Aggiungi la cartella che contiene TMFG_core.py al sys.path
sys.path.append(
    os.path.abspath(
        os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    )
)

# Ora puoi importare la classe TMFG e rinominarla
from TMFG_core import *  # TMFG as fast_tmfg


import numpy as np


data = np.random.randint(0, 100, size=(100, 50))
corr = np.square(np.corrcoef(data, rowvar=False))
cov = np.cov(data, rowvar=False)
model = TMFG()

# output='logo'
cliques, seps, adj_matrix = model.fit_transform(weights=corr, cov=cov, output="logo")
print(type(adj_matrix))
print(adj_matrix.shape)
print(adj_matrix)

# output = 'unweighted_sparse_W_matrix'
cliques, seps, adj_matrix = model.fit_transform(
    weights=corr, cov=cov, output="unweighted_sparse_W_matrix"
)
print(type(adj_matrix))
print(adj_matrix.shape)
print(adj_matrix)

# output = 'weighted_sparse_W_matrix'
cliques, seps, adj_matrix = model.fit_transform(
    weights=corr, cov=cov, output="weighted_sparse_W_matrix"
)
print(type(adj_matrix))
print(adj_matrix.shape)
print(adj_matrix)
