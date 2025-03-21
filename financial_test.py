import numpy as np
import matplotlib.pyplot as plt

import pyRMT as rmt
import yfinance as yf
import pandas as pd
import time
import os
import argparse
import traceback
from scipy.linalg import logm, inv


def load_stock_data(file_name="returns_data_1060.csv"):

    df = pd.read_csv(
        file_name, index_col=0
    )  # index_col=0 usa la prima colonna come indice (es. date)

    # metto a 0 i NaN
    df.fillna(0, inplace=True)

    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data = data.T
    X_train, X_test = data[:, :1000], data[:, 1000:]

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


def optimal_weights(Sigma, g):
    """
    Compute the optimal weights for the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    g (numpy.ndarray): Predictions vector for the portfolio.

    Returns:
    w (numpy.ndarray): Optimal weights for the portfolio.
    """

    w = np.dot(inv(Sigma), g) / np.dot(g, np.dot(inv(Sigma), g))

    return w


def variance_portfolio(Sigma, w):
    """
    Compute the variance of the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    w (numpy.ndarray): Weights of the portfolio.

    Returns:
    var (float): Variance of the portfolio.
    """

    var = np.dot(w, np.dot(Sigma, w))

    return var


def portfolio_statistics(X_train, Sigma=np.identity(500), strategy="min_var"):
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

    if strategy == "min_var":
        g = minimum_variance_portfolio_vector(X_train)
    elif strategy == "omn":
        Oracle_train, Oracle_test = load_stock_data(
            file_name="oracle_returns_data_1060.csv"
        )
        g = omniscient_portfolio_vector(Oracle_train)
    elif strategy == "mean_rev":
        g = mean_reversion_portfolio_vector(X_train)
    elif strategy == "rnd":
        g = random_long_short_portfolio_vector(X_train)

    # toy example

    w = optimal_weights(Sigma, g)
    var = variance_portfolio(Sigma, w)

    #print(f"Strategy: {strategy}, portfolio variance: {var}")

    return var


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
    E_Clipped = rmt.clipped(X_train, alpha=0.2, return_covariance=True)

    # Compute the statistics of the portfolio
    # Create a dictionary mapping Sigmas to their corresponding methods
    Sigma_methods = {
        "Sample": E_sample,
        "Rie": E_rie,
        "IW": E_iw,
        "Clipped": E_Clipped
    }
    strategies = ["min_var", "omn", "mean_rev", "rnd"]

    print("STRATEGY  |  METHOD  |  VARIANCE")
    print("---------------------------------")
    for strategy in strategies:
        for method, Sigma in Sigma_methods.items():
            var = portfolio_statistics(X_train, Sigma, strategy=strategy)
            print(f"{strategy}  |  {method}  | {var}")

def check():

    # Load the stock data
    X_train, X_test = load_stock_data()
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
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")


if __name__ == "__main__":
    check()
    main()
