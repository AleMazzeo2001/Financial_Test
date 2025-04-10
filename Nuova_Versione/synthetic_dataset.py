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


# Check Standardization
def check_standardization():
    X_train, X_test = FCA.load_stock_data(file_name="log_returns_data_1060.csv")
    X_train_std, std_daily, std_stocks = FCA.standardize_returns(X_train)
    diff = X_train - X_train_std

    # print("Media colonne dopo standardizzazione:", np.mean(X_train_std, axis=0))
    print("Media righe dopo standardizzazione:", np.mean(X_train_std, axis=1))
    # print("Deviazione standard colonne:", np.std(X_train_std, axis=0))
    print("Deviazione standard righe:", np.std(X_train_std, axis=1))
    print(f"X_train:{np.linalg.norm(np.mean(X_train, axis=1))}")
    print(f"X_std:{np.linalg.norm(np.mean(X_train_std, axis=1))}")

    time = np.arange(1, 801, 1)
    plt.plot(time, X_train[0, :], label="dati")
    plt.plot(time, X_train[1, :], label="dati")
    plt.plot(time, X_train[2, :], label="dati")
    plt.title("Dati non standardizzati")
    plt.legend()
    plt.show()

    plt.plot(time, X_train_std[0, :])
    plt.plot(time, X_train_std[1, :])
    plt.plot(time, X_train_std[2, :])
    plt.title("Dati standardizzati")
    plt.show()


np.random.seed(27029)

# define parameters
N_Stocks = 400
T_train = 800
T_out = 60
n_sets = 50
T_tot = n_sets * T_out

X_Train_Real, X_test_Real = FCA.load_stock_data(file_name="returns_data_1060.csv")

# C_true =  np.diag(np.arange(1, 401))
#A = 2 * np.random.randn(N_Stocks, N_Stocks) + 5 * np.ones([N_Stocks, N_Stocks])
#C_true = (A @ A.T) / N_Stocks
#C_true = np.eye(N_Stocks)
#C_true = np.diag(np.arange(1, N_Stocks + 1))
C_true= np.cov(X_Train_Real)


mean = np.zeros(N_Stocks)
# print(C_true.shape)

simulated_data_train = FCA.generate_dataset(C_true, T_train, 1, type="Student")
simulated_data_test = FCA.generate_dataset(C_true, T_out, n_sets, type="Student")

# print(simulated_data_train[0,:])
simulated_data_train = np.squeeze(simulated_data_train)

def sorted_eigenvalues(matrix, std_stocks=None):
    """Sort eigenvalues in descending order."""
    eigenvalues = np.linalg.eigvals(matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    if std_stocks is not None:
        std_stocks = np.squeeze(std_stocks)
        sorted_eigenvalues = sorted_eigenvalues * (std_stocks**2)

    return sorted_eigenvalues


def get_mutual_info_matrix_continuous(data: np.ndarray) -> np.ndarray:
    n_features = data.shape[1]
    mi_matrix = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Calcola la MI tra la feature j (input) e la feature i (target)
            mi_ij = mutual_info_regression(
                data[:, [j]], data[:, i], discrete_features=False
            )[0]
            mi_ji = mutual_info_regression(
                data[:, [i]], data[:, j], discrete_features=False
            )[0]
            value = (mi_ij + mi_ji) / 2  # Simmetrizza

            mi_matrix[i, j] = value
            mi_matrix[j, i] = value

    return mi_matrix


def plot_eigenvalues(data, C_true, show=False):
    data_std, std_daily, std_stocks = FCA.standardize_returns(data)
    print(std_stocks.shape)

    E_Sample = np.cov(data)
    E_Sample_std = np.cov(data_std)
    E_Sample_std_Grossa = (std_stocks @ std_stocks.T) @ E_Sample_std
    E_rie = rmt.optimalShrinkage(data, return_covariance=True, method="rie")
    E_rie = (std_stocks @ std_stocks.T) @ E_rie
    E_iw = rmt.optimalShrinkage(data, return_covariance=True, method="iw")
    E_iw = (std_stocks @ std_stocks.T) @ E_iw
    E_Clipped = rmt.clipped(data, alpha=0.0, return_covariance=True)
    E_Clipped = (std_stocks @ std_stocks.T) @ E_Clipped

    model = TMFG()
    corr = np.square(np.corrcoef(data, rowvar=True))
    E_Sample_TMFG = np.cov(data)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    # TMFG with mutual information
    # mi_matrix = get_mutual_info_matrix_continuous(data)
    E_Sample_MI = np.cov(data)
    MI=np.load("MI_Matrix.npy")
    _, _, J_MI = model.fit_transform(weights=MI, cov=E_Sample_MI, output="logo")
    E_MI = np.linalg.inv(J_MI)

    # Autovalori della matrice di covarianza vera
    lambda_C = sorted_eigenvalues(C_true)

    # Autovalori della matrice di covarianza stimata
    lambda_E_Sample = sorted_eigenvalues(E_Sample)
    lambda_E_Sample_std = sorted_eigenvalues(E_Sample_std)
    lambda_E_Sample_std_Grossa = sorted_eigenvalues(E_Sample_std_Grossa)
    lambda_E_rie = sorted_eigenvalues(E_rie, std_stocks=None)
    lambda_E_iw = sorted_eigenvalues(E_iw, std_stocks=None)
    lambda_E_Clipped = sorted_eigenvalues(E_Clipped, std_stocks=None)
    lambda_E_TMFG = sorted_eigenvalues(E_TMFG, std_stocks=None)
    lambda_E_MI = sorted_eigenvalues(E_MI, std_stocks=None)

    # Autovalori della matrice di precisione vera
    J = np.linalg.inv(C_true)
    J_Sample = np.linalg.inv(E_Sample)
    J_Sample_std = np.linalg.inv(E_Sample_std)
    J_Sample_std_Grossa = np.linalg.inv(E_Sample_std_Grossa)
    J_rie = np.linalg.inv(E_rie)
    J_iw = np.linalg.inv(E_iw)
    J_Clipped = np.linalg.inv(E_Clipped)

    lambda_J = sorted_eigenvalues(J)

    # Autovalori della matrice di precisione stimata
    lambda_E_Sample_J = sorted_eigenvalues(J_Sample)
    lambda_E_Sample_std_J = sorted_eigenvalues(J_Sample_std)
    lambda_E_Sample_std_Grossa_J = sorted_eigenvalues(J_Sample_std_Grossa)
    lambda_E_rie_J = sorted_eigenvalues(J_rie)
    lambda_E_iw_J = sorted_eigenvalues(J_iw)
    lambda_E_Clipped_J = sorted_eigenvalues(J_Clipped)
    lambda_J_TMFG = sorted_eigenvalues(J_TMFG)
    lambda_J_MI = sorted_eigenvalues(J_MI)

    # Figura con due subplot affiancati
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Primo subplot: Covarianza
    axs[0].plot(lambda_C, lambda_C, label="C_true")
    axs[0].plot(lambda_C, lambda_E_Sample, label="E_Sample")
    # axs[0].plot(lambda_C, lambda_E_Sample_std, label="E_Sample_std")
    # axs[0].plot(lambda_C, lambda_E_Sample_std_Grossa, label="E_Sample_std_Grossa")
    axs[0].plot(lambda_C, lambda_E_rie, label="E_Rie")
    axs[0].plot(lambda_C, lambda_E_iw, label="E_IW")
    axs[0].plot(lambda_C, lambda_E_Clipped, label="E_Clipped")
    axs[0].plot(lambda_C, lambda_E_TMFG, label="E_TMFG")
    axs[0].plot(lambda_C, lambda_E_MI, label="E_MI")
    axs[0].set_xlabel("True Eigenvalues")
    axs[0].set_ylabel("Reconstructed Eigenvalues")
    axs[0].set_title("Covariance Matrix")
    axs[0].legend()

    # Secondo subplot: Precision Matrix
    axs[1].plot(lambda_J, lambda_J, label="J_true")
    axs[1].plot(lambda_J, lambda_E_Sample_J, label="E_Sample")
    # axs[1].plot(lambda_J, lambda_E_Sample_std_J, label="E_Sample_std")
    # axs[1].plot(lambda_C, lambda_E_Sample_std_Grossa_J, label="E_Sample_std_Grossa")
    axs[1].plot(lambda_J, lambda_E_rie_J, label="E_Rie")
    axs[1].plot(lambda_J, lambda_E_iw_J, label="E_IW")
    axs[1].plot(lambda_J, lambda_E_Clipped_J, label="E_Clipped")
    axs[1].plot(lambda_J, lambda_J_TMFG, label="J_TMFG")
    axs[1].plot(lambda_J, lambda_J_MI, label="J_MI")
    axs[1].set_xlabel("True Eigenvalues")
    axs[1].set_ylabel("Reconstructed Eigenvalues")
    axs[1].set_title("Precision Matrix")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    if show:
        fig, axs = plt.subplots(2, 3, figsize=(12, 10))

        # Plot C_true
        im0 = axs[0, 0].imshow(C_true, aspect="auto")
        axs[0, 0].set_title("True Covariance Matrix")
        fig.colorbar(im0, ax=axs[0, 0])

        # Plot E_Sample
        im1 = axs[0, 1].imshow(E_Sample, aspect="auto")
        axs[0, 1].set_title("Sample Covariance Matrix")
        fig.colorbar(im1, ax=axs[0, 1])

        # Plot E_iw
        im4 = axs[0, 2].imshow(E_iw, aspect="auto")
        axs[0, 2].set_title("IW Covariance Matrix")
        fig.colorbar(im4, ax=axs[0, 2])

        # Plot E_Rie
        im2 = axs[1, 0].imshow(E_rie, aspect="auto")
        axs[1, 0].set_title("RIE Covariance Matrix")
        fig.colorbar(im2, ax=axs[1, 0])

        # Plot E_Clipped
        im3 = axs[1, 1].imshow(E_Clipped, aspect="auto")
        axs[1, 1].set_title("Clipped Covariance Matrix")
        fig.colorbar(im3, ax=axs[1, 1])

        # Plot E_TMFG
        im5 = axs[1, 2].imshow(E_TMFG, aspect="auto")
        axs[1, 2].set_title("TMFG Covariance Matrix")
        fig.colorbar(im5, ax=axs[1, 2])

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 3, figsize=(12, 10))

        # Plot J
        im0 = axs[0, 0].imshow(J, aspect="auto")
        axs[0, 0].set_title("True Precision Matrix")
        fig.colorbar(im0, ax=axs[0, 0])

        # Plot E_Sample
        im1 = axs[0, 1].imshow(J_Sample, aspect="auto")
        axs[0, 1].set_title("Sample Precision Matrix")
        fig.colorbar(im1, ax=axs[0, 1])

        # Plot E_iw
        im4 = axs[0, 2].imshow(J_iw, aspect="auto")
        axs[0, 2].set_title("IW Precision Matrix")
        fig.colorbar(im4, ax=axs[0, 2])

        # Plot E_Rie
        im2 = axs[1, 0].imshow(J_rie, aspect="auto")
        axs[1, 0].set_title("RIE Precision Matrix")
        fig.colorbar(im2, ax=axs[1, 0])

        # Plot E_Clipped
        im3 = axs[1, 1].imshow(J_Clipped, aspect="auto")
        axs[1, 1].set_title("Clipped Precision Matrix")
        fig.colorbar(im3, ax=axs[1, 1])

        # Plot E_TMFG
        im5 = axs[1, 2].imshow(J_TMFG, aspect="auto")
        axs[1, 2].set_title("TMFG Precision Matrix")
        fig.colorbar(im5, ax=axs[1, 2])

        plt.tight_layout()
        plt.show()


plot_eigenvalues(simulated_data_train, C_true, show=False)


FCA.Compute_Performances(simulated_data_train, simulated_data_test, OUTPUT="Multiple_Boxplot", Compute_MI=False)
