import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_selection import mutual_info_regression

import sys
import os


tmfg_core_path = os.path.expanduser('~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph')
mfcf_path = os.path.expanduser('~/Desktop/UCL/CODE/MFCF')
FCA_path = os.path.expanduser('/Users/alessandromazzeo/Desktop/UCL/CODE/Nuova_Versione')
sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)
sys.path.append(FCA_path)
import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table
import financial_test as FCA



def Compute_J_MFCF(C, Min_Cl=5, Max_Cl=5, Coordination_Number=2, threshold=0.01, drop_sep=True, return_J=False):
    """
    Compute the MFCF(Min_Cl, Max_Cl, Coordination_Number) Network.

    Parameters:
    C (numpy.ndarray): Correlation matrix.(Mutual information matrix???)
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




np.random.seed(seed=27029)
X_train_emp, X_test_emp = FCA.load_stock_data(file_name="~/Desktop/UCL/CODE/Nuova_Versione/returns_data_1060.csv")
N, T_train = X_train_emp.shape
T_out=60
n_sets=65
C_true= np.cov(X_train_emp)


X = FCA.generate_dataset(C_true, T_train, 1, type="Gaussian")
X = np.squeeze(X)
simulated_data_test = FCA.generate_dataset(C_true, T_out, n_sets, type="Gaussian")
C = np.corrcoef(X, rowvar=True)

Compute_MI=False
if Compute_MI==True:
    MI = mutual_info_matrix(X)
    np.save("MI_Matrix.npy", MI)
else:
    MI = np.load("MI_Matrix.npy")

J_True = np.linalg.inv(C_true)

#TMFG with two different classes
print("Correlation Matrix")
J1, _, _, _, _ = Compute_J_MFCF(C, 4, 4, 1, return_J=True)
model = tmfg.TMFG()
E_Sample_TMFG = np.cov(X)
_, _, J_TMFG = model.fit_transform(weights=C, cov=E_Sample_TMFG, output="logo")

print(f"J_True - J1: {np.linalg.norm(J_True - J1):.2e} " )
print(f"J_True - J_TMFG: {np.linalg.norm(J_True - J_TMFG):.2e}")
print(f"J1 - J_TMFG: {np.linalg.norm(J1 - J_TMFG):.2e}" )



#Mutual INformation
print("Mutual Information")
J1, _, _, _, _ = Compute_J_MFCF(MI, 4, 4, 1, return_J=True)
model = tmfg.TMFG()
E_Sample_TMFG = np.cov(X)
_, _, J_TMFG = model.fit_transform(weights=MI, cov=E_Sample_TMFG, output="logo")

print(f"J_True - J1: {np.linalg.norm(J_True - J1):.2e}" )
print(f"J_True - J_TMFG: {np.linalg.norm(J_True - J_TMFG):.2e}" )
print(f"J1 - J_TMFG: {np.linalg.norm(J1 - J_TMFG):.2e} ")

show=True
if show:
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))

    # Plot C_true
    im0 = axs[ 0].imshow(J_True, aspect="auto")
    axs[ 0].set_title("True Precision Matrix")
    fig.colorbar(im0, ax=axs[ 0])

    # Plot J_MFCF
    im1 = axs[ 1].imshow(J1, aspect="auto")
    axs[ 1].set_title("MFCF Precision Matrix")
    fig.colorbar(im1, ax=axs[ 1])

    # Plot J_TMFG
    im2 = axs[ 2].imshow(J_TMFG, aspect="auto")
    axs[ 2].set_title("TMFG Precision Matrix")
    fig.colorbar(im1, ax=axs[ 2])
    
    plt.tight_layout()
    plt.show()



