import numpy as np
import matplotlib.pyplot as plt

import pyRMT as rmt
import yfinance as yf
import pandas as pd
import time
import os
import argparse
import traceback
from scipy.linalg import logm, inv, pinv

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    )
)
from TMFG_core import *  # TMFG as fast_tmfg


def load_stock_data(file_name="returns_data_1060.csv"):

    df = pd.read_csv(
        file_name, index_col=0
    )  # index_col=0 usa la prima colonna come indice (es. date)

    # metto a 0 i NaN
    # df.fillna(0, inplace=True)

    if df.isna().values.any():
        # Conta quanti NaN ci sono
        nan_count = df.isna().sum().sum()
        print(f"Numero totale di NaN nel DataFrame: {nan_count}")

    # Sostituisci i NaN con interpolazione lineare
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)  # Backfill per riempire NaN alle estremità
    df.ffill(inplace=True)  # Forward fill per riempire NaN alle estremità

    # df.display()

    # Controlla se ci sono ancora NaN e sostituiscili con 0
    if df.isnull().values.any():
        print("Attenzione: ci sono ancora NaN nei dati. Verranno sostituiti con 0.")
        nan_count = df.isnull().sum().sum()
        print(f"Numero totale di NaN nel DataFrame: {nan_count}")

        df.fillna(0, inplace=True)

    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data = data.T
    # X_train, X_test = data[:, :1000], data[:, 1000:]
    X_train, X_test = (
        data[:400, :800],
        data[:400, 800:860],
    )  # ho modificato i valori per tenere q=cost
    return X_train, X_test


def standardize_returns(R):
    """
    Standardize the returns matrix R.

    Parameters:
    R (numpy.ndarray): Matrix of returns.

    Returns:
    X (numpy.ndarray): Matrix of standardized returns.
    """
    N, T = R.shape
    std_dev = np.std(R, axis=1, keepdims=True)
    std_dev[std_dev == 0] = 1  # Evita la divisione per zero
    X = (R - np.mean(R, axis=1, keepdims=True)) / std_dev
    std_dev_col = np.std(X, axis=0, keepdims=True)
    std_dev_col[std_dev_col == 0] = (
        1  # Evita la divisione per zero nella seconda dimensione
    )
    X = X / std_dev_col

    return X


def minimum_variance_portfolio_vector(X):
    """
    Compute the predictions vector  g for the minimum variance portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.

    Returns:
    g (numpy.ndarray): Predictions vector for the minimum variance portfolio.
    """

    N, T = X.shape
    g = np.ones(N)

    return g


def omniscient_portfolio_vector(O_train):
    """
    Compute the predictions vector  g for the omniscient portfolio.

    Parameters:
    O_train (numpy.ndarray): Matrix of oracle returns with T_out=60. (google colab for dataset creation)


    Returns:
    g (numpy.ndarray): Predictions vector for the omniscient portfolio.
    """

    N, T = O_train.shape
    std_dev = np.std(O_train, axis=1)

    # Check if there are rows with zero standard deviation
    zero_std_rows = np.where(std_dev == 0)[0]
    if len(zero_std_rows) > 0:
        print(
            f"Attenzione! Le seguenti righe hanno deviazione standard zero: {zero_std_rows}"
        )

    std_dev[std_dev == 0] = 1  # Evita la divisione per zero
    g = np.sqrt(N) * O_train[:, 0] / std_dev

    return g


def mean_reversion_portfolio_vector(X_train):
    """
    Compute the predictions vector  g for the mean reversion portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.
    T_out (int): Number of days for the prediction.

    Returns:
    g (numpy.ndarray): Predictions vector for the mean reversion portfolio.
    """

    N, T = X_train.shape
    g = -np.sqrt(N) * X_train[:, 0]
    return g


def random_long_short_portfolio_vector(X):
    """
    Compute the predictions vector  g for the random long-short portfolio made by indipendent standardize gaussian variables
    with unitary norm.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.

    Returns:
    g (numpy.ndarray): Predictions vector for the random long-short portfolio.
    """

    N, T = X.shape
    vec = np.random.randn(N)
    vec = vec / np.linalg.norm(vec)
    g = np.sqrt(N) * vec

    return g


def optimal_weights(Sigma, g, J_Precision=None):
    """
    Compute the optimal weights for the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    g (numpy.ndarray): Predictions vector for the portfolio.

    Returns:
    w (numpy.ndarray): Optimal weights for the portfolio.
    """

    # Check if Sigma is singular or has very small eigenvalues => use the pseudo-inverse
    if np.linalg.det(Sigma) == 0:
        print("Attenzione! Sigma è singolare.")
        w = np.dot(pinv(Sigma), g) / np.dot(g, np.dot(pinv(Sigma), g))
    eigvals = np.linalg.eigvals(Sigma)
    if np.min(np.abs(eigvals)) < 1e-10:
        print("Attenzione! Sigma ha autovalori molto piccoli.")
        w = np.dot(pinv(Sigma), g) / np.dot(g, np.dot(pinv(Sigma), g))

    else:
        w = np.dot(inv(Sigma), g) / np.dot(g, np.dot(inv(Sigma), g))

    if J_Precision is not None:
        w = np.dot(J_Precision, g) / np.dot(g, np.dot(J_Precision, g))

    return w


def variance_portfolio(Sigma, w, J_Precision=None):
    """
    Compute the variance of the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    w (numpy.ndarray): Weights of the portfolio.

    Returns:
    var (float): Variance of the portfolio.
    """

    var = np.dot(w, np.dot(Sigma, w))

    if J_Precision is not None:
        var = np.dot(w, np.dot(inv(J_Precision), w))
    return var


def portfolio_statistics(
    X_train, Sigma=np.identity(500), strategy="min_var", J_Precision=None
):
    """
    Compute the statistics of the portfolio.

    Parameters:
    X_train (numpy.ndarray): Matrix of standardized returns.

    Sigma (numpy.ndarray): Covariance matrix of the returns.
        Default is the identity matrix (Isotropic Case).

    strategy (str): Strategy for the portfolio.
        min_var -> Minimum variance portfolio.
        omn -> Omniscient portfolio.
        mean_rev -> Mean reversion portfolio.
        rnd -> Random long-short portfolio.

    Returns:
    mean (float): Mean of the portfolio.
    var (float): Variance of the portfolio.
    """

    if strategy == "min_var ":
        g = minimum_variance_portfolio_vector(X_train)
    elif strategy == "omn     ":
        Oracle_train, Oracle_test = load_stock_data(
            file_name="oracle_returns_data_1060.csv"
        )
        g = omniscient_portfolio_vector(Oracle_train)
    elif strategy == "mean_rev":
        g = mean_reversion_portfolio_vector(X_train)
    elif strategy == "rnd     ":
        g = random_long_short_portfolio_vector(X_train)

    if J_Precision is not None:
        w = optimal_weights(Sigma, g, J_Precision)
        var = variance_portfolio(Sigma, w, J_Precision)
    else:
        w = optimal_weights(Sigma, g)
        var = variance_portfolio(Sigma, w)

    # print(f"Strategy: {strategy}, portfolio variance: {var}")

    return var


def plot_eigenvalues(E0, E1, E2, E3, E_Clipped, Oracle=None):
    """
    Plotta gli autovalori delle matrici ripulite rispetto agli autovalori campionari.
    """
    eig_E0 = np.linalg.eigvalsh(E0)
    eig_E1 = np.linalg.eigvalsh(E1)
    eig_E2 = np.linalg.eigvalsh(E2)

    eig_E3 = np.linalg.eigvalsh(E3)
    eig_clip = np.linalg.eigvalsh(E_Clipped)

    plt.figure(figsize=(8, 6))
    plt.plot(eig_E0, eig_E1, "o", label="RIE", alpha=0.7)
    plt.plot(eig_E0, eig_E2, "s", label="IW", alpha=0.7)
    plt.plot(eig_E0, eig_E3, "d", label="Kernel", alpha=0.7)
    plt.plot(eig_E0, eig_clip, "x", label="Clipped", alpha=0.7)

    if Oracle is not None:
        eig_Oracle = np.linalg.eigvalsh(Oracle)
        plt.plot(eig_E0, eig_Oracle, "o", label="Oracle", alpha=0.7)

    plt.xlabel("Autovalori campionari")
    plt.ylabel("Autovalori ripuliti")
    plt.legend()
    plt.title("Autovalori ripuliti vs. Autovalori campionari")
    plt.show()


def main():

    # define parameters
    N = 500
    T = 1000
    T_out = 60

    # Load the stock data
    X_train, X_test = load_stock_data()

    # Standardize the returns
    X_train = standardize_returns(X_train)
    X_test = standardize_returns(X_test)
    Oracle_train, Oracle_test = load_stock_data(
        file_name="oracle_returns_data_1060.csv"
    )

    E_sample = np.cov(X_train)
    E_rie = rmt.optimalShrinkage(X_train, return_covariance=False, method="rie")
    E_iw = rmt.optimalShrinkage(X_train, return_covariance=False, method="iw")
    E_Clipped = rmt.clipped(X_train, alpha=0.2, return_covariance=False)
    # TMFG
    model = TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=False))
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_sample, output="logo")

    # Plot the eigenvalues
    # plot_eigenvalues(E_sample, E_rie, E_iw, E_sample, E_Clipped)

    # Compute the statistics of the portfolio
    # Create a dictionary mapping Sigmas to their corresponding methods
    Sigma_methods = {
        "Sample ": E_sample,
        "Rie    ": E_rie,
        "IW     ": E_iw,
        "Clipped": E_Clipped,
        "TMFG   ": J_TMFG,
    }
    strategies = ["min_var ", "omn     ", "mean_rev", "rnd     "]

    print("STRATEGY  |  METHOD  |  VARIANCE")
    print("----------------------------------------------")
    for strategy in strategies:
        for method, Sigma in Sigma_methods.items():

            if method == "TMFG   ":
                var = portfolio_statistics(
                    X_train, Sigma, strategy=strategy, J_Precision=Sigma
                )
            else:
                var = portfolio_statistics(X_train, Sigma, strategy=strategy)
            print(f"{strategy}  |  {method}  | {var}")
        print("----------------------------------------------")


def check():

    # Load the stock data
    X_train, X_test = load_stock_data()
    print("Dimensioni Train:", X_train.shape)
    print("Dimensioni Test:", X_test.shape)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")

    # Standardize the returns
    X_train = standardize_returns(X_train)
    X_test = standardize_returns(X_test)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train dopo la standardizzazione!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test dopo la standardizzazione!")

    # Oracle data
    Oracle_train, Oracle_test = load_stock_data(
        file_name="oracle_returns_data_1060.csv"
    )
    print("Dimensioni Oracle Train:", Oracle_train.shape)
    print("Dimensioni Oracle Test:", Oracle_test.shape)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")


if __name__ == "__main__":
    # check()
    main()
