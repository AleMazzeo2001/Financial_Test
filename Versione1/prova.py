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
#print(type(adj_matrix))
#print(adj_matrix.shape)
#print(adj_matrix)

# output = 'unweighted_sparse_W_matrix'
cliques, seps, adj_matrix = model.fit_transform(
    weights=corr, cov=cov, output="unweighted_sparse_W_matrix"
)

# output = 'weighted_sparse_W_matrix'
cliques, seps, adj_matrix = model.fit_transform(
    weights=corr, cov=cov, output="weighted_sparse_W_matrix"
)



x = np.arange(0, 100, dtype=int)
print(x)

# Creazione delle finestre di test consecutive non overlapping
test_data_list = []
start_idx = 0
T_out=5
num_timesteps = x.shape[0]


    # Size Check
    # print(f"num_timesteps: {num_timesteps}, train_size: {train_size}, T_out: {T_out}")
    # print(f"Condizione iniziale: {train_size + T_out} <= {num_timesteps} -> {train_size + T_out <= num_timesteps}")

while start_idx + T_out <= num_timesteps:
        # print(f"Aggiungo dati da {start_idx} a {start_idx + T_out}")  # Debug
        test_data_list.append(x[start_idx : start_idx + T_out])
        start_idx += T_out  # Finestra non overlapping

    
if len(test_data_list) == 0:
    raise ValueError(
        "Errore: nessun dato disponibile per il test. Controlla i parametri di input."
    )
#print(f"test_data_list: {test_data_list}")  # Debug

test_righe, test_colonne = np.shape(test_data_list)
for i in range(test_righe):
    print(test_data_list[i])  # Debug