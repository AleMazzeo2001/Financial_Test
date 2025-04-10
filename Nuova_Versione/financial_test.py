import numpy as np
import matplotlib.pyplot as plt
import pyRMT as rmt
import yfinance as yf
import pandas as pd
import time
import os
import argparse
import traceback
from scipy.linalg import logm, inv, pinv, LinAlgError
from scipy.stats import multivariate_t
from sklearn.covariance import shrunk_covariance
from sklearn.feature_selection import mutual_info_regression

import sys
import os

tmfg_core_path = os.path.expanduser(
    "~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph"
)
mfcf_path = os.path.expanduser("~/Desktop/UCL/CODE/MFCF")
FCA_path = os.path.expanduser("/Users/alessandromazzeo/Desktop/UCL/CODE/Nuova_Versione")
sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)
sys.path.append(FCA_path)
import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table
import financial_test as FCA

path = "/Users/alessandromazzeo/Desktop/UCL/CODE/"


def load_stock_data(
    file_name=path + "returns_data_1060.csv", train_size=800, N_stocks=400, T_out=60
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
        X = R

        """
        std_dev_daily = np.std(X, axis=0, keepdims=True, ddof=0) * np.sqrt(T)  # Paper Bouchaud
        #print(f"std_dev_daily: {std_dev_daily.shape}")
        if np.any(std_dev_daily <1e-10 ):
            print(
                "Attenzione! Ci sono righe con deviazione standard zero nel training."
            )
            std_dev_daily[std_dev_daily == 0] = 1  # Evita la divisione per zero
        
        std_dev_stocks = np.std(X, axis=1, keepdims=True, ddof=0)
        if np.any(std_dev_stocks <1e-10):
            print(
                "Attenzione! Ci sono colonne con deviazione standard zero nel training."
            )
            std_dev_stocks[std_dev_stocks == 0] = (1 )
        
        X = (X-np.mean(X, axis=1, keepdims=True)) /( std_dev_stocks *std_dev_daily)
        """
        # Standardizzazione sulle colonne (asse 0)
        col_mean = np.mean(X, axis=0, keepdims=True)
        std_daily = np.std(X, axis=0, keepdims=True, ddof=0)  # *np.sqrt(T)
        std_daily[std_daily < 1e-10] = 1  # Evita divisioni per zero
        X = (X) / std_daily

        # Standardizzazione sulle righe (asse 1)
        row_mean = np.mean(X, axis=1, keepdims=True)
        std_stocks = np.std(X, axis=1, keepdims=True, ddof=0)
        std_stocks[std_stocks < 1e-10] = 1  # Evita divisioni per zero
        X = (X - row_mean) / std_stocks

    else:
        n_series, N, T = R.shape
        X = np.zeros((n_series, N, T))

        for i in range(n_series):
            X_temp = R[i]
            std_daily = np.std(X_temp, axis=0, keepdims=True, ddof=0) * np.sqrt(T)
            if np.any(std_daily < 1e-10):
                print(
                    f"Attenzione! Ci sono righe con deviazione standard zero in serie {i}."
                )
                std_daily[std_daily == 0] = 1

            std_stocks = np.std(X_temp, axis=1, keepdims=True, ddof=0)
            if np.any(std_stocks < 1e-10):
                print(
                    f"Attenzione! Ci sono colonne con deviazione standard zero in serie {i}."
                )
                std_stocks[std_stocks == 0] = 1
            X[i] = (X_temp - np.mean(X_temp, axis=1, keepdims=True)) / (
                std_stocks * std_daily
            )

    return X, std_daily, std_stocks


def standardize_returns_2(X_train, X_test):
    """
    unisce X_train e X_test in un'unica matrice, standardizza tutto assieme e poi ti ritorna
    solo X_test_std
    """
    N, T_train = X_train.shape
    n_series, N_test, T_test = X_test.shape

    if N != N_test:
        raise ("Il numero di Stocks non coindice.")

    # 1. Rimodellare X_test
    # Permutiamo gli assi per portare la dimensione 400 come prima dimensione
    X_test_reshaped = np.transpose(X_test, (1, 0, 2))  # Ora ha forma (400, 65, 60)
    X_test_reshaped = X_test_reshaped.reshape(N, -1)  # Rimodelliamo in (400, 65 * 60)

    # 2. Concatenare con X_train
    # X_combined = np.concatenate((X_train, X_test_reshaped), axis=1)
    X_combined = X_test_reshaped

    # 3. Standardizzare le righe
    mean = np.mean(X_combined, axis=1, keepdims=True)
    std_stocks = np.std(X_combined, axis=1, keepdims=True)
    std_stocks[std_stocks == 0] = 1
    std_daily = np.std(X_combined, axis=0, keepdims=True) * np.sqrt(T_test * n_series)
    std_daily[std_daily == 0] = 1
    X_combined_standardized = (X_combined - mean) / (std_stocks * std_daily)

    # 4. Separare nuovamente le matrici
    # X_train_standardized = X_combined_standardized[:, :800]
    # Estraiamo X_test_standardized e rimodelliamo alla forma originale
    X_test_standardized = X_combined_standardized[:,].reshape(N, n_series, T_test)
    # Trasponiamo gli assi per riportare alla forma originale (65, 400, 60)
    X_test_standardized = np.transpose(X_test_standardized, (1, 0, 2))

    return X_test_standardized, std_daily, std_stocks


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


def omniscient_portfolio_vector(O_train_std):
    """
    Compute the predictions vector  g for the omniscient portfolio.

    Parameters:
    O_train (numpy.ndarray): Matrix of oracle returns with T_out=60. (google colab for dataset creation)


    Returns:
    g (numpy.ndarray): Predictions vector for the omniscient portfolio.
    """

    N, T = O_train_std.shape
    """
    O_train_std, _, _= standardize_returns(O_train)
    std_dev = np.std(O_train, axis=1)

    # Check if there are rows with zero standard deviation
    zero_std_rows = np.where(std_dev == 1)[0]
    if len(zero_std_rows) > 0:
        print(
            f"Attenzione! Le seguenti righe hanno deviazione standard zero: {zero_std_rows}"
        )
        std_dev[std_dev == 0] = 1
    # print(f"O_train.shape, {O_train[:,0].shape}")

    g = np.sqrt(N) * O_train[:, 0] / std_dev"""
    g = np.sqrt(N) * O_train_std[:, 0]
    return g


def mean_reversion_portfolio_vector(O_train_std):
    """
    Compute the predictions vector  g for the mean reversion portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.
    T_out (int): Number of days for the prediction.

    Returns:
    g (numpy.ndarray): Predictions vector for the mean reversion portfolio.
    """

    N, T = O_train_std.shape
    # O_train_std, _, _= standardize_returns(O_train_std)
    """std_dev = np.std(O_train, axis=1)

    # Check if there are rows with zero standard deviation
    zero_std_rows = np.where(std_dev == 1)[0]
    if len(zero_std_rows) > 0:
        print(
            f"Attenzione! Le seguenti righe hanno deviazione standard zero: {zero_std_rows}"
        )
        std_dev[std_dev == 0] = 1
    # print(f"O_train.shape, {O_train[:,0].shape}")

    g = -np.sqrt(N) * O_train[:, 0] / std_dev"""
    g = -np.sqrt(N) * O_train_std[:, 0]
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


def optimal_weight(
    Sigma, g, std_daily=None, std_stocks=None, J_Precision=None
):  # hai tolto la s finale, versione vecchia
    """
    Compute the optimal weights for the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    g (numpy.ndarray): Predictions vector for the portfolio.
    std_daily (numpy.ndarray): Daily volatilities (1 x T).
    std_stocks (numpy.ndarray): Stock volatilities (N x 1).
    J_Precision (numpy.ndarray or None): Optional precision matrix.

    Returns:
    w (numpy.ndarray): Optimal weights for the portfolio.
    """

    # Tentativo 1: se Sigma ha problemi numerici (es. autovalori piccoli), uso pinv
    try:
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(np.abs(eigvals)) < 1e-10:
            raise LinAlgError("Autovalori troppo piccoli, uso pseudo-inversa.")

        if J_Precision is not None:
            w = J_Precision @ g / (g @ J_Precision @ g)

        elif std_stocks is None:
            Sigma_inv = inv(Sigma)
            w = Sigma_inv @ g / (g @ Sigma_inv @ g)
        else:
            Sigma_rescaled = (std_stocks @ std_stocks.T) @ Sigma
            Sigma_inv = inv(Sigma_rescaled)
            w = Sigma_inv @ g / (g @ Sigma_inv @ g)

    except LinAlgError as e:
        print(f"⚠️ Errore numerico: {e}. Uso la pseudo-inversa.")
        Sigma_pinv = pinv(Sigma)
        w = Sigma_pinv @ g / (g @ Sigma_pinv @ g)

    return w


def optimal_weights(Sigma, g, std_daily=None, std_stocks=None, J_Precision=None):
    """
    Compute the optimal weights for the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    g (numpy.ndarray): Predictions vector for the portfolio.
    std_daily (numpy.ndarray): Daily volatilities (1 x T).
    std_stocks (numpy.ndarray): Stock volatilities (N x 1).
    J_Precision (numpy.ndarray or None): Optional precision matrix.

    Returns:
    w (numpy.ndarray): Optimal weights for the portfolio.
    """
    try:
        if J_Precision is not None:
            # Se ho già la precision matrix, la uso direttamente
            w = J_Precision @ g / (g @ J_Precision @ g)

        elif std_stocks is None:
            # Controllo la condizione della matrice Sigma solo se la sto usando
            eigvals = np.linalg.eigvalsh(Sigma)
            if np.min(np.abs(eigvals)) < 1e-10:
                raise np.linalg.LinAlgError("Autovalori troppo piccoli in Sigma.")

            Sigma_inv = np.linalg.inv(Sigma)
            w = Sigma_inv @ g / (g @ Sigma_inv @ g)

        else:
            Sigma_rescaled = (std_stocks @ std_stocks.T) * Sigma
            eigvals = np.linalg.eigvalsh(Sigma_rescaled)
            if np.min(np.abs(eigvals)) < 1e-10:
                raise np.linalg.LinAlgError(
                    "Autovalori troppo piccoli in Sigma_rescaled."
                )

            Sigma_inv = np.linalg.inv(Sigma_rescaled)
            w = Sigma_inv @ g / (g @ Sigma_inv @ g)

    except np.linalg.LinAlgError as e:
        print(f"⚠️ Errore numerico: {e} Uso la pseudo-inversa.")

        if std_stocks is None:
            Sigma_pinv = np.linalg.pinv(Sigma)
        else:
            Sigma_rescaled = (std_stocks @ std_stocks.T) * Sigma
            Sigma_pinv = np.linalg.pinv(Sigma_rescaled)

        w = Sigma_pinv @ g / (g @ Sigma_pinv @ g)

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
    X_train, Sigma, std_daily, std_stocks, strategy="min_var", J_Precision=None
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
        w = optimal_weights(Sigma, g, std_daily, std_stocks, J_Precision=J_Precision)
        var = variance_portfolio(Sigma, w)
    else:
        w = optimal_weights(Sigma, g, std_daily, std_stocks)
        var = variance_portfolio(Sigma, w)

    # print(f"Strategy: {strategy}, portfolio variance: {var}")

    return g, w, var


def is_positive_definite(matrix):
    # Calcola gli autovalori della matrice
    eigenvalues = np.linalg.eigvals(matrix)

    # Verifica se tutti gli autovalori sono positivi
    return np.all(eigenvalues > 0)


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


def generate_dataset(C, T, n_sets, type="Student", df=3):
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

    if type == "Gaussian":
        # Genera dati da una distribuzione normale multivariata
        for i in range(n_sets):
            data[i] = np.random.multivariate_normal(mean=np.zeros(N), cov=C, size=T).T
    elif type == "Student":
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


def Compute_Performances(
    X_train, X_test, Oracle_Train_=None, Oracle_Test_=None, OUTPUT=None, Compute_MI=False
):
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot
    Compute_MI:
        -True: calcola e salva la matrice della Mutua Informazione
        _FAlse: carica la matrice della mutua informazione
    
    """

    # Standardize the returns
    X_train_std, std_daily, std_stocks = standardize_returns(X_train)
    X_test_std, _, _ = standardize_returns_2(X_train, X_test)

    path = "/Users/alessandromazzeo/Desktop/UCL/CODE/"
    Oracle_train, Oracle_test = load_stock_data(
        file_name="oracle_returns_data_1060.csv"
    )

    if Oracle_Train_ is not None:
        Oracle_train = Oracle_Train_
    if Oracle_Test_ is not None:
        Oracle_test = Oracle_Test_

    Oracle_test_std, _, _ = standardize_returns(Oracle_test)

    E_sample = np.cov(X_train)
    E_rie = rmt.optimalShrinkage(X_train, return_covariance=True, method="rie")
    E_iw = rmt.optimalShrinkage(X_train, return_covariance=True, method="iw")
    E_Clipped = rmt.clipped(X_train, alpha=0.0, return_covariance=True)
    E_shrunk = shrunk_covariance(E_sample, shrinkage=0.1)

    # TMFG
    model = tmfg.TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=True))
    E_Sample_TMFG = np.cov(X_train)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    # TMFG + MI (Mutual Information)
    if Compute_MI == True:
        MI=mutual_info_matrix(X_train)
        np.save('MI_Matrix.npy', MI)
    if Compute_MI == False:
        MI=np.load('MI_Matrix.npy')

    _, _, J_TMFG_MI = model.fit_transform(weights=MI, cov=E_Sample_TMFG, output="logo")
    E_TMFG_MI = np.linalg.inv(J_TMFG_MI)


    # Check if the matrices are positive definite
    if not is_positive_definite(E_sample):
        raise ValueError("Sample covariance matrix not positive definite!")
    if not is_positive_definite(E_rie):
        raise ValueError("RIE covariance matrix not positive definite!")
    if not is_positive_definite(E_iw):
        raise ValueError("IW covariance matrix not positive definite!")
    if not is_positive_definite(E_Clipped):
        raise ValueError("Clipped covariance matrix not positive definite!")
    if not is_positive_definite(E_shrunk):
        raise ValueError("Shrunk covariance matrix not positive definite!")
    if not is_positive_definite(J_TMFG):
        raise ValueError("Precision TMFG matrix not positive definite!")
    if not is_positive_definite(E_TMFG):
        raise ValueError("TMFG covariance matrix not positive definite!")
    if not is_positive_definite(J_TMFG_MI):
        raise ValueError("Precision TMFG_MI matrix not positive definite!")
    if not is_positive_definite(E_TMFG_MI):
        raise ValueError("TMFG_MI covariance matrix not positive definite!")
    

    # Compute the statistics of the portfolio

    Sigma_methods = {
        "Sample ": E_sample,
        "Rie    ": E_rie,
        "IW     ": E_iw,
        "Clipped": E_Clipped,
        "Shrunk ": E_shrunk,
        "TMFG   ": (E_TMFG, J_TMFG),
        "TMFG_MI": (E_TMFG_MI, J_TMFG_MI),

    }

    n_methods = len(Sigma_methods)

    strategies = ["min_var ", "omn     ", "mean_rev", "rnd     "]

    print("\nIN SAMPLE CASE\n")
    Optimal_Weights_dict = {}

    print("STRATEGY  |  METHOD  |  VARIANCE")
    print("----------------------------------------------")
    for strategy in strategies:
        for method, Sigma in Sigma_methods.items():

            if method == "Sample ":
                g, w, var = portfolio_statistics(
                    X_train,
                    Sigma,
                    std_daily=None,
                    std_stocks=None,
                    strategy=strategy,
                )
            elif method == "TMFG   ":
                E_TMFG, J_TMFG = Sigma
                g, w, var = portfolio_statistics(
                    X_train,
                    E_TMFG,
                    std_daily,
                    std_stocks,
                    strategy=strategy,
                    J_Precision=J_TMFG,
                )
            elif method == "TMFG_MI":
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
                    X_train,
                    Sigma,
                    std_daily=None,
                    std_stocks=None,
                    strategy=strategy,
                )

            Optimal_Weights_dict[(strategy, method)] = w

            print(f"{strategy}  |  {method}  | {var:.3e}")
        print("----------------------------------------------")

    print("\nOUT OF SAMPLE CASE\n")
    print("STRATEGY  |  METHOD  |  VARIANCE              |  CV%")
    print("-------------------------------------------------------")

    index = 0
    variance_data = {}
    for (strategy, method), w in Optimal_Weights_dict.items():

        # if strategy == "omn     ":
        #    TEST_DATA = Oracle_test_std
        if method == "TMFG   ":
            TEST_DATA = X_test_std
        else:
            TEST_DATA = X_test_std

        performances = np.zeros(TEST_DATA.shape[0])

        for i in range(TEST_DATA.shape[0]):

            var = Risk_Out(TEST_DATA[i], w)
            performances[i] = var

        mean_var = np.mean(performances)
        std_var = np.std(
            performances, ddof=1
        )  # Use ddof=1 for sample standard deviation
        variance_data[(strategy, method)] = performances
        print(
            f"{strategy}  |  {method}  | {mean_var:.2e} +/- {std_var:.1e}  | {round((100*std_var)/np.abs(mean_var), 1)}"
        )

        index += 1
        if (index % n_methods) == 0:
            print("-------------------------------------------------------")

    if OUTPUT is not None:
        Show_Outliers(variance_data, OUTPUT=OUTPUT)


if __name__ == "__main__":
    pass
