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
from scipy.stats import multivariate_t

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    )
)
from TMFG_core import *


def load_stock_data(
    file_name="returns_data_1060.csv", train_size=800, N_stocks=400, T_out=60
):
    """
    Carica i dati delle azioni e li suddivide in training e test set.

    Parameters:
    - data: array 2D (shape: num_timesteps x num_assets), contenente i prezzi o rendimenti.
    - train_size: numero di dati da usare per il training (default: 800).
    - T_out: lunghezza delle finestre di test (default: 60).

    Returns:
    - train_data: primi `train_size` dati.
    - test_data_list: lista di array di shape (T_out, num_assets), con blocchi non overlapping.
    """

    df = pd.read_csv(
        file_name, index_col=0
    )  # index_col=0 usa la prima colonna come indice (es. date)

    # Chek for NaN values
    if df.isna().values.any():
        nan_count = df.isna().sum().sum()
        print(f"Total number of NaN values in DataFrame: {nan_count}")

    # Linear interpolation for NaN values
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)  # Backfill per riempire NaN alle estremità
    df.ffill(inplace=True)  # Forward fill per riempire NaN alle estremità

    # Check for NaN values again => fill with 0
    if df.isnull().values.any():
        print("Attenzione: ci sono ancora NaN nei dati. Verranno sostituiti con 0.")
        nan_count = df.isnull().sum().sum()
        print(f"Total number of NaN values in DataFrame: {nan_count}")
        df.fillna(0, inplace=True)

    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data = data.T  # TxN => NxT

    num_assets, num_timesteps = data.shape

    train_data = data[:N_stocks, :train_size]
    test_data = data[:N_stocks, train_size:]

    # Creazione delle finestre di test consecutive non overlapping
    test_data_list = []
    start_idx = train_size

    # Size Check
    # print(f"num_timesteps: {num_timesteps}, train_size: {train_size}, T_out: {T_out}")
    # print(f"Condizione iniziale: {train_size + T_out} <= {num_timesteps} -> {train_size + T_out <= num_timesteps}")

    while start_idx + T_out <= num_timesteps:
        test_data_list.append(data[:N_stocks, start_idx : start_idx + T_out])
        start_idx += T_out  # Finestra non overlapping

    if len(test_data_list) == 0:
        raise ValueError(
            "Errore: nessun dato disponibile per il test. Controlla i parametri di input."
        )

    # Converti la lista in un array NumPy
    test_data = np.stack(test_data_list)

    return train_data, test_data


def standardize_returns(R):
    """
    Standardize the returns matrix R.

    Parameters:
    R (numpy.ndarray): Matrix of returns.

    Returns:
    X (numpy.ndarray): Matrix of standardized returns.
    std_dev_daily (numpy.ndarray T): Standard deviation of each colum.
    std_dev_stocks (numpy.ndarray N): Standard deviation of each row.

    If R is 2D, it returns:
    - X: standardized returns matrix

    If R is 3D, it returns:
    - X: 3D matrix of standardized returns
    """
    if R.ndim == 2:
        N, T = R.shape
        X = R - np.mean(R, axis=1, keepdims=True)
        std_dev_daily = np.std(X, axis=0, keepdims=True, ddof=0) * np.sqrt(
            T
        )  # Paper Bouchaud
        print(f"std_dev_daily: {std_dev_daily.shape}")
        if np.any(std_dev_daily == 0):
            print(
                "Attenzione! Ci sono righe con deviazione standard zero nel training."
            )
            std_dev_daily[std_dev_daily == 0] = 1  # Evita la divisione per zero
        X = X / std_dev_daily
        print(f"X: {X.shape}")
        std_dev_stocks = np.std(X, axis=1, keepdims=True, ddof=0)
        if np.any(std_dev_stocks == 0):
            print(
                "Attenzione! Ci sono colonne con deviazione standard zero nel training."
            )
            std_dev_stocks[std_dev_stocks == 0] = (
                1  # Evita la divisione per zero nella seconda dimensione
            )
        print(f"std_dev_stocks: {std_dev_stocks.shape}")
        X = X / std_dev_stocks
        print(f"X: {X.shape}")

    else:
        n_series, N, T = R.shape
        X = np.zeros((n_series, N, T))

        for i in range(n_series):
            X_temp = R[i] - np.mean(R[i], axis=1, keepdims=True)
            std_dev_daily = np.std(X_temp, axis=0, keepdims=True, ddof=0) * np.sqrt(T)
            if np.any(std_dev_daily == 0):
                print(
                    f"Attenzione! Ci sono righe con deviazione standard zero in serie {i}."
                )
                std_dev_daily[std_dev_daily == 0] = 1
            X_temp = X_temp / std_dev_daily
            std_dev_stocks = np.std(X_temp, axis=1, keepdims=True, ddof=0)
            if np.any(std_dev_stocks == 0):
                print(
                    f"Attenzione! Ci sono colonne con deviazione standard zero in serie {i}."
                )
                std_dev_stocks[std_dev_stocks == 0] = 1
            X[i] = X_temp / std_dev_stocks

    return X, std_dev_daily, std_dev_stocks


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
    zero_std_rows = np.where(std_dev == 1)[0]
    if len(zero_std_rows) > 0:
        print(
            f"Attenzione! Le seguenti righe hanno deviazione standard zero: {zero_std_rows}"
        )
        std_dev[std_dev == 0] = 1
    # print(f"O_train.shape, {O_train[:,0].shape}")

    g = np.sqrt(N) * O_train[:, 0] / std_dev
    return g


def mean_reversion_portfolio_vector(O_train):
    """
    Compute the predictions vector  g for the mean reversion portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.
    T_out (int): Number of days for the prediction.

    Returns:
    g (numpy.ndarray): Predictions vector for the mean reversion portfolio.
    """

    N, T = O_train.shape
    std_dev = np.std(O_train, axis=1)

    # Check if there are rows with zero standard deviation
    zero_std_rows = np.where(std_dev == 1)[0]
    if len(zero_std_rows) > 0:
        print(
            f"Attenzione! Le seguenti righe hanno deviazione standard zero: {zero_std_rows}"
        )
        std_dev[std_dev == 0] = 1
    # print(f"O_train.shape, {O_train[:,0].shape}")

    g = -np.sqrt(N) * O_train[:, 0] / std_dev
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


def check_norm(g):
    """
    Check if the norm of the vector g is equal to sqrt(N).

    Parameters:
    g (numpy.ndarray): Predictions vector.

    Returns:
    g (numpy.ndarray): Normalized predictions vector.
    """
    N = g.shape[0]

    norm = np.linalg.norm(g)
    if np.abs(norm - np.sqrt(N)) < 1e-4:
        pass
    else:
        print(f"Norma del vettore g non è uguale a sqrt(N): {norm}")
        c = np.sqrt(N) / norm
        g = g * c
    return g


def optimal_weights(Sigma, g, std_daily, std_stocks, J_Precision=None):
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
        print("Attenzione! Sigma è singolare => Pseudo_Inverse")
        w = np.dot(pinv(Sigma), g) / np.dot(g, np.dot(np.linalg.pinv(Sigma), g))

    eigvals = np.linalg.eigvals(Sigma)

    Sigma = (std_stocks @ std_stocks.T) @ Sigma

    if np.min(np.abs(eigvals)) < 1e-10:
        print("Attenzione! Sigma ha autovalori molto piccoli.")
        w = np.dot(pinv(Sigma), g) / np.dot(g, np.dot(pinv(Sigma), g))

    if J_Precision is not None:
        J_Precision = J_Precision  # / (std * std.T)
        w = np.dot(J_Precision, g) / np.dot(g, np.dot(J_Precision, g))

    else:
        w = np.dot(inv(Sigma), g) / np.dot(g, np.dot(inv(Sigma), g))

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

    if var < 0:
        raise ValueError("Portfolio variance is negative!")

    return var


def portfolio_statistics(
    X_train, Sigma, std, std_col, strategy="min_var", J_Precision=None
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
    # Portfolio vector
    if strategy == "min_var ":
        g = minimum_variance_portfolio_vector(X_train)
    elif strategy == "omn     ":
        Oracle_train, Oracle_test = load_stock_data(
            file_name="oracle_returns_data_1060.csv"
        )
        g = omniscient_portfolio_vector(Oracle_train)
    elif strategy == "mean_rev":
        Oracle_train, Oracle_test = load_stock_data(
            file_name="oracle_returns_data_1060.csv"
        )
        g = mean_reversion_portfolio_vector(Oracle_train)
    elif strategy == "rnd     ":
        g = random_long_short_portfolio_vector(X_train)

    g = check_norm(g)
    print(f"Norma del vettore g: {np.linalg.norm(g)}")

    # Optimal weights
    if J_Precision is not None:
        w = optimal_weights(Sigma, g, std, std_col, J_Precision=J_Precision)
        var = variance_portfolio(Sigma, w)
    else:
        w = optimal_weights(Sigma, g, std, std_col)
        var = variance_portfolio(Sigma, w)

    # print(f"Strategy: {strategy}, portfolio variance: {var}")

    return g, w, var


def is_positive_definite(matrix):
    # Calcola gli autovalori della matrice
    eigenvalues = np.linalg.eigvals(matrix)

    # Verifica se tutti gli autovalori sono positivi
    return np.all(eigenvalues > 0)


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


def Risk_Out(X, w):
    """
    Compute the out-of-sample risk of the portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.
    w (numpy.ndarray): Weights of the portfolio.

    Returns:
    out_risk (float): Out-of-sample risk of the portfolio.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array!")

    Sigma = np.cov(X)
    var = variance_portfolio(Sigma, w)

    return var


def generate_dataset(C, T, n_sets, df=3):
    """
    Genera un tensore 3D contenente n_sets di dati estratti da una distribuzione di Student multivariata.

    Parametri:
    C      -- matrice di covarianza (NxN)
    T      -- numero di campioni per ogni set
    n_sets -- numero di set indipendenti da generare
    df     -- gradi di libertà della distribuzione di Student (default: 3)

    Ritorna:
    Un array (n_sets, N, T) di dati campionati.
    """
    np.random.seed(27029)  # Per la riproducibilità

    N = C.shape[0]  # Dimensione della matrice di covarianza
    data = np.zeros((n_sets, N, T))  # Preallocazione del tensore

    for i in range(n_sets):
        data[i] = multivariate_t.rvs(
            loc=np.zeros(N), shape=C, df=df, size=T
        ).T  # Trasposta per avere (N, T)

    return data


def Show_Outliers(variance_data, OUTPUT="Single_Boxplot"):
    """
    Show the outliers of the variance data using boxplots.

    Parameters:
    - variance_data: Dictionary containing the variance data for each strategy and method.
       variance_data[(strategy, method)] = performances
    - OUTPUT: String indicating the type of boxplot to create.
        "Single_Boxplot" for a single boxplot for each strategy and method.
        "Multiple_Boxplot" for multiple boxplots for each strategy.

    Returns:
    - None
    """

    if OUTPUT == "Single_Boxplot":
        for (strategy, methods), var in variance_data.items():
            plt.figure(figsize=(8, 6))
            plt.boxplot(var)
            plt.title(f"Box Plot per {strategy}, {methods}")
            plt.ylabel("Varianza")
            plt.xlabel("Metodo")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()

    elif OUTPUT == "Multiple_Boxplot":
        grouped_data = {}
        for (strategy, methods), var in variance_data.items():
            if strategy not in grouped_data:
                grouped_data[strategy] = {}
            grouped_data[strategy][methods] = var

        # Creiamo i boxplot per ogni strategia
        for strategy, methods_data in grouped_data.items():
            plt.figure(figsize=(8, 6))  # Creiamo una figura per ogni strategia

            # Estraiamo i dati per ogni metodo
            data = list(methods_data.values())
            labels = list(methods_data.keys())

            plt.boxplot(data, labels=labels)  # Creiamo il boxplot per tutti i metodi
            plt.title(f"Box Plot per {strategy}")
            plt.ylabel("Varianza")
            plt.xlabel("Metodo")
            plt.xticks(rotation=45)  # Ruotiamo le etichette se sono lunghe
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.show()


def Compute_Performances(X_train, X_test, OUTPUT=None):
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot
    """

    # Standardize the returns
    X_train_std, std_daily, std_stocks = standardize_returns(X_train)
    # X_test = standardize_returns(X_test)
    Oracle_train, Oracle_test = load_stock_data(
        file_name="oracle_returns_data_1060.csv"
    )

    E_sample = np.cov(X_train_std)
    E_rie = rmt.optimalShrinkage(X_train, return_covariance=False, method="rie")
    E_iw = rmt.optimalShrinkage(X_train, return_covariance=False, method="iw")
    E_Clipped = rmt.clipped(X_train, alpha=0.0, return_covariance=False)

    # TMFG
    model = TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=True))
    E_Sample_TMFG = np.cov(X_train)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    # Check if the matrices are positive definite
    if not is_positive_definite(E_sample):
        raise ValueError("Sample covariance matrix not positive definite!")
    if not is_positive_definite(E_rie):
        raise ValueError("RIE covariance matrix not positive definite!")
    if not is_positive_definite(E_iw):
        raise ValueError("IW covariance matrix not positive definite!")
    if not is_positive_definite(E_Clipped):
        raise ValueError("Clipped covariance matrix not positive definite!")
    if not is_positive_definite(J_TMFG):
        raise ValueError("Precision TMFG matrix not positive definite!")
    if not is_positive_definite(E_TMFG):
        raise ValueError("TMFG covariance matrix not positive definite!")

    # Plot the eigenvalues
    # plot_eigenvalues(E_sample, E_rie, E_iw, E_sample, E_Clipped)

    # Compute the statistics of the portfolio

    # Create a dictionary mapping Sigmas to their corresponding methods
    Sigma_methods = {
        "Sample ": E_sample,
        "Rie    ": E_rie,
        "IW     ": E_iw,
        "Clipped": E_Clipped,
        "TMFG   ": (E_TMFG, J_TMFG),
    }

    strategies = ["min_var ", "omn     ", "mean_rev", "rnd     "]

    print("\nIN SAMPLE CASE\n")
    Optimal_Weights_dict = {}

    print("STRATEGY  |  METHOD  |  VARIANCE")
    print("----------------------------------------------")
    for strategy in strategies:
        for method, Sigma in Sigma_methods.items():

            if method == "TMFG   ":
                E_TMFG, J_TMFG = Sigma
                g, w, var = portfolio_statistics(
                    X_train,
                    E_TMFG,
                    std_daily,
                    std_stocks,
                    strategy=strategy,
                    J_Precision=J_TMFG,
                )
            else:
                g, w, var = portfolio_statistics(
                    X_train, Sigma, std_daily, std_stocks, strategy=strategy
                )

            Optimal_Weights_dict[(strategy, method)] = w

            print(f"{strategy}  |  {method}  | {var:.3e}")
        print("----------------------------------------------")

    print("\nOUT OF SAMPLE CASE\n")
    print("STRATEGY  |  METHOD  |  VARIANCE")
    print("-----------------------------------------------")

    index = 0
    variance_data = {}
    for (strategy, method), w in Optimal_Weights_dict.items():

        if strategy == "omn     ":
            TEST_DATA = Oracle_test
        else:
            TEST_DATA = X_test

        performances = np.zeros(TEST_DATA.shape[0])

        for i in range(TEST_DATA.shape[0]):

            var = Risk_Out(TEST_DATA[i], w)
            performances[i] = var

        mean_var = np.mean(performances)
        std_var = np.std(
            performances, ddof=1
        )  # Use ddof=1 for sample standard deviation
        variance_data[(strategy, method)] = performances
        print(f"{strategy}  |  {method}  | {mean_var:.2e} +/- {std_var:.1e}")

        index += 1
        if (index % 5) == 0:
            print("-----------------------------------------------")

    if OUTPUT is not None:
        Show_Outliers(variance_data, OUTPUT=OUTPUT)


def main():

    # define parameters
    N = 400
    T = 800
    T_out = 60

    # Load the stock data
    X_train, X_test = load_stock_data()

    Compute_Performances(X_train, X_test, OUTPUT=None)


def check():

    print("\n\n--CHECK DATA--\n\n")

    # Load the stock data
    X_train, X_test = load_stock_data()
    print("Dimensioni Train:", X_train.shape)
    print("Dimensioni Test:", X_test.shape)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")

    # Standardize the returns
    X_train, std, std_col = standardize_returns(X_train)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train dopo la standardizzazione!")

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


def simulated_data():

    print("\n\n--SIMULATED DATA--\n\n")

    # define parameters
    T_train = 800
    T_out = 60
    n_sets = 50
    T_tot = n_sets * T_out

    # Load the stock data
    X_train, X_test = load_stock_data()
    X_train_std, std_daily, std_stocks = standardize_returns(X_train)

    C_true = np.cov(X_train_std)

    # Generate simulated data
    simulated_data_train = generate_dataset(C_true, T_train, 1)
    simulated_data_train = np.squeeze(simulated_data_train)
    simulated_data_test = generate_dataset(C_true, T_out, n_sets)

    print(f"Dimensioni Dati Simulati Train", simulated_data_train.shape)
    # print(f"Dimensioni Dati Simulati Test", simulated_data_test.shape)
    print(f"Dimensioni Covarianza", C_true.shape)

    # Compute_Performances(simulated_data_train, simulated_data_test, OUTPUT="Single_Boxplot")

    # Standardize the returns
    X_train_std, std_daily, std_stocks = standardize_returns(simulated_data_train)

    E_sample = np.cov(X_train_std)
    E_rie = rmt.optimalShrinkage(X_train_std, return_covariance=True, method="rie")
    E_iw = rmt.optimalShrinkage(X_train_std, return_covariance=True, method="iw")
    E_Clipped = rmt.clipped(X_train_std, alpha=0.0, return_covariance=True)

    # TMFG
    model = TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=True))
    E_Sample_TMFG = np.cov(X_train_std)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    # Calcolo degli autovalori e ordinamento in ordine decrescente
    lambda_C = np.sort(np.linalg.eigvals(C_true))[::-1]
    lambda_E_sample = np.sort(np.linalg.eigvals(E_sample))[::-1]
    lambda_E_rie = np.sort(np.linalg.eigvals(E_rie))[::-1]
    lambda_E_iw = np.sort(np.linalg.eigvals(E_iw))[::-1]
    lambda_E_Clipped = np.sort(np.linalg.eigvals(E_Clipped))[::-1]
    lambda_E_TMFG = np.sort(np.linalg.eigvals(E_TMFG))[::-1]

    plt.plot(lambda_C, lambda_C, marker="o", linestyle="-", label="C_True")
    plt.plot(lambda_C, lambda_E_sample, marker="o", linestyle="-", label="E_Sample")
    plt.plot(lambda_C, lambda_E_rie, marker="o", linestyle="-", label="E_Rie")
    plt.plot(lambda_C, lambda_E_iw, marker="o", linestyle="-", label="E_IW")
    plt.plot(lambda_C, lambda_E_Clipped, marker="o", linestyle="-", label="E_Clipped")
    plt.plot(lambda_C, lambda_E_TMFG, marker="o", linestyle="-", label="E_TMFG")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    check()
    # simulated_data()
    main()
