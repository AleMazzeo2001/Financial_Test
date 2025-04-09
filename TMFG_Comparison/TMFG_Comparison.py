import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_selection import mutual_info_regression

import sys
import os

# Percorsi dei file
tmfg_core_path = os.path.expanduser('~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph')
mfcf_path = os.path.expanduser('~/Desktop/UCL/CODE/MFCF')

# Aggiungi i percorsi a sys.path
sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)

# Ora puoi importare i moduli
import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table



def Compute_J_MFCF(Min_Cl=5, Max_Cl=5, Coordination_Number=2, threshold=0.01, drop_sep=True, return_J=False):
    """
    Compute the MFCF(Min_Cl, Max_Cl, Coordination_Number) Network.
    Parameters:
    Min_Cl (int): Minimum clique size.
    Max_Cl (int): Maximum clique size.
    Coordination_Number (int): Coordination number.
    threshold (float): Threshold for clique extension.
    drop_sep (bool): Whether to drop separators.
    return_J (bool): Whether to return the J matrix.
    Returns:
    cliques (list): List of cliques.
    separators (list): List of separators.
    peo (list): Perfect elimination order.
    gt (gain_table): Gain table.
    J (numpy.ndarray): J matrix if return_J is True.

    if return_J:
        return J, cliques, separators, peo, gt
    else:
        return cliques, separators, peo, gt

    """
    ctl = mfcf.mfcf_control()
    ctl['min_clique_size'] = Min_Cl
    ctl['max_clique_size'] = Max_Cl
    ctl['coordination_number'] = Coordination_Number
    ctl['threshold'] = threshold
    ctl['drop_sep'] = drop_sep

    gain_function = gf.sumsquares_gen
    cliques, separators, peo, gt = mfcf.mfcf(C, ctl, gain_function)

    if return_J:
        J = mfcf.logo(C, cliques, separators)
        return J, cliques, separators, peo, gt
    else:
        return cliques, separators, peo, gt


def mutual_info_matrix(data):
    """
    Calcola la matrice della mutua informazione per un dataset di variabili continue.

    Parametri:
    data (numpy.ndarray): Matrice 2D con shape (n_variabili, n_osservazioni).

    Ritorna:
    numpy.ndarray: Matrice quadrata (n_variabili x n_variabili) della mutua informazione.
    """
    # Trasponi la matrice per avere le variabili come colonne
    data_t = data.T
    n_variabili = data_t.shape[1]
    mi_matrix = np.zeros((n_variabili, n_variabili))

    for i in range(n_variabili):
        for j in range(i, n_variabili):
            if i == j:
                mi = mutual_info_regression(data_t[:, i].reshape(-1, 1), data_t[:, i])[0]
            else:
                mi = mutual_info_regression(data_t[:, i].reshape(-1, 1), data_t[:, j])[0]
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # La matrice è simmetrica

    return mi_matrix


p = 15
T = 100

np.random.seed(seed=27029)
X = np.random.normal(0,1,(T,p)).T
print(X.shape)

C = np.corrcoef(X, rowvar=True)
print(C.shape)

MI=mutual_info_matrix(X)
print(MI.shape)
