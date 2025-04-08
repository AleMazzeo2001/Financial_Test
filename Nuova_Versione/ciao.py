import numpy as np
import matplotlib.pyplot as plt
import financial_test as FCA
from scipy.linalg import logm, inv, pinv, LinAlgError
import pyRMT as rmt
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    )
)
from TMFG_core import *

N_Stocks = 400
T_train = 800
T_out = 60
n_sets = 50
T_tot = n_sets * T_out


X_Train_Real, X_test_Real = FCA.load_stock_data(file_name="returns_data_1060.csv")
C_true= np.cov(X_Train_Real)

simulated_data_train = FCA.generate_dataset(C_true, T_train, 1, type="Gaussian")
simulated_data_train=np.squeeze(simulated_data_train)
simulated_data_test = FCA.generate_dataset(C_true, T_out, n_sets, type="Gaussian")

E_Sample = np.cov(simulated_data_train)

def inverted_matrix(Sigma):
    try:
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(np.abs(eigvals)) < 1e-10:
            raise np.linalg.LinAlgError("Autovalori troppo piccoli in Sigma.")
            
        Sigma_inv = np.linalg.inv(Sigma)

    except np.linalg.LinAlgError as e:
        print(f"⚠️ Errore numerico: {e} Uso la pseudo-inversa.")
        
        Sigma_inv = np.linalg.pinv(Sigma)

    return Sigma_inv

J_Sample=inverted_matrix(E_Sample)
J_Sample_Cazzuta=np.linalg.inv(E_Sample)
diff = np.linalg.norm(J_Sample - J_Sample_Cazzuta)
print(f"Differenza tra le matrici inverse: {diff:.2e}")


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot C_true
im0 = axs[0,0].imshow(J_Sample_Cazzuta, aspect="auto")
axs[0, 0].set_title("Cazzuta Inverse Matrix")
fig.colorbar(im0, ax=axs[0])

# Plot E_Sample
im1 = axs[0, 1].imshow(J_Sample, aspect="auto")
axs[0, 1].set_title("Inverse Sample Covariance Matrix")
fig.colorbar(im1, ax=axs[0, 1])
plt.tight_layout()
plt.title(f"Difference: {diff:.2f}")

im3 = axs[1,0].imshow(J_Sample-J_Sample_Cazzuta, aspect="auto")
axs[1, 0].set_title("DiffMatrix")
fig.colorbar(im3, ax=axs[1,0])
plt.show()