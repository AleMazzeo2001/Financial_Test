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
from scipy.stats import kendalltau
from sklearn.covariance import shrunk_covariance
from sklearn.feature_selection import mutual_info_regression
import shutil

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

import pickle




def load_stock_data_rolling(
    file_name="returns_data_1060.csv", train_size=800, N_stocks=400, T_out=60, len_rolling=100
):
    """
    Carica i dati delle azioni e genera coppie rolling (training, test) non sovrapposte.

    Parameters:
    - file_name: nome del file CSV contenente i rendimenti (shape: time x assets)
    - train_size: numero di timestep per ciascun blocco di training
    - N_stocks: numero di asset da considerare
    - T_out: numero di timestep per ciascun blocco di test
    - len_rolling: lunghezza del rolling

    Returns:
    - train_data: array di shape (n_windows, N_stocks, train_size)
    - test_data: array di shape (n_windows, N_stocks, T_out)
    """

    df = pd.read_csv(file_name, index_col=0)

    # Trattamento dei NaN
    if df.isna().values.any():
        print(f"Total number of NaN values before interpolation: {df.isna().sum().sum()}")
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    if df.isna().values.any():
        print("Ancora NaN trovati dopo il riempimento, verranno sostituiti con 0.")
        df.fillna(0, inplace=True)

    # Conversione in NumPy array e trasposizione
    data = df.to_numpy().T  # shape: (N_assets, T)
    data = data[:N_stocks, :]  # Seleziona i primi N_stocks

    num_assets, num_timesteps = data.shape

    # Generazione finestre rolling
    train_windows = []
    test_windows = []

    start_idx = 0
    while start_idx + train_size + T_out <= num_timesteps:
        train_block = data[:, start_idx : start_idx + train_size]
        test_block = data[:, start_idx + train_size : start_idx + train_size + T_out]

        train_windows.append(train_block)
        test_windows.append(test_block)

        start_idx += len_rolling  # Rolling di un solo timestep

    if len(train_windows) == 0:
        raise ValueError("Nessuna finestra valida trovata: controlla la lunghezza dei dati o i parametri train_size/T_out.")

    # Stack in array 3D
    train_data = np.stack(train_windows)  # shape: (n_windows, N_stocks, train_size)
    test_data = np.stack(test_windows)    # shape: (n_windows, N_stocks, T_out)

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

    return X, std_daily, std_stocks


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
            Sigma_pinv = np.linalg.inv(Sigma)
        else:
            Sigma_rescaled = (std_stocks @ std_stocks.T) * Sigma
            Sigma_pinv = np.linalg.inv(Sigma_rescaled)

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

    #if var < 0:
    #    raise ValueError("Portfolio variance is negative!")

    return var


def portfolio_statistics(
    X_train, Sigma, Oracle_train, Oracle_test, std_daily, std_stocks, strategy="min_var", J_Precision=None
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
        g = omniscient_portfolio_vector(Oracle_train)
    elif strategy == "mean_rev":
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

    print(f"Strategy: {strategy}, portfolio variance: {var}")

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
    Show the outliers of the variance data and computation times using boxplots.

    Parameters:
    - variance_data: Dictionary containing the variance data and computation times
                     for each strategy and method:
                     rolling_performance_dict[(strategy, method)] = {
                         'risks': [...],
                         'time': [...]
                     }
    - OUTPUT: String indicating the type of boxplot to create.
        "Single_Boxplot" for a single boxplot per (strategy, method)
        "Multiple_Boxplot" for multiple boxplots grouped by strategy

    Returns:
    - None
    """
    import matplotlib.pyplot as plt

    if OUTPUT == "Single_Boxplot":
        for (strategy, method), result in variance_data.items():
            risks = result["risks"]
            times = result["time"]

            # Boxplot varianza
            plt.figure(figsize=(8, 6))
            plt.boxplot(risks)
            plt.title(f"Box Plot della VARIANZA per {strategy}, {method}")
            plt.ylabel("Varianza")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()

            # Boxplot tempo
            plt.figure(figsize=(8, 6))
            plt.boxplot(times)
            plt.title(f"Box Plot del TEMPO DI CALCOLO per {strategy}, {method}")
            plt.ylabel("Tempo (s)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()

    elif OUTPUT == "Multiple_Boxplot":
        grouped_risks = {}
        grouped_times = {}

        for (strategy, method), result in variance_data.items():
            risks = result["risks"]
            times = result["time"]

            if strategy not in grouped_risks:
                grouped_risks[strategy] = {}
                grouped_times[strategy] = {}

            grouped_risks[strategy][method] = risks
            grouped_times[strategy][method] = times

        # Creiamo due boxplot per ogni strategia: uno per la varianza e uno per i tempi
        for strategy in grouped_risks:
            methods = list(grouped_risks[strategy].keys())

            # Boxplot varianza
            data_risks = [grouped_risks[strategy][m] for m in methods]
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_risks, labels=methods)
            plt.title(f"Box Plot della VARIANZA per {strategy}")
            plt.ylabel("Varianza")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()

            # Boxplot tempo
            data_times = [grouped_times[strategy][m] for m in methods]
            plt.figure(figsize=(10, 6))
            plt.boxplot(data_times, labels=methods)
            plt.title(f"Box Plot del TEMPO DI CALCOLO per {strategy}")
            plt.ylabel("Tempo (s)")
            plt.xticks(rotation=45)
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



def kendall_tau_matrix(data):
    """
    Calcola la matrice di Kendall's rank correlation (Tau) per una matrice NxT,
    dove N è il numero di variabili (righe) e T il numero di osservazioni (colonne).

    Parameters:
    - data: np.ndarray, shape (N, T)

    Returns:
    - tau_matrix: np.ndarray, shape (N, N), simmetrica con valori di Tau
    """
    N = data.shape[0]
    tau_matrix = np.ones((N, N))

    for i in range(N):
        for j in range(i+1, N):
            tau, _ = kendalltau(data[i], data[j])
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau  # simmetrica

    return tau_matrix


from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import kendalltau
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import eigh

def kendall_tau_matrix_parallel(data, max_workers=None, clean_psd=False):
    """
    Compute a cleaned covariance matrix using Kendall Tau rank correlations.

    Parameters:
    - data: np.ndarray of shape (N_assets, T_time)
    - max_workers: optional, number of parallel threads
    - clean_psd: if True, projects the Pearson matrix to the nearest PSD

    Returns:
    - cov_matrix: np.ndarray of shape (N_assets, N_assets)
    """

    N = data.shape[0]
    tau_matrix = np.ones((N, N))

    def compute_tau(i, j):
        tau, _ = kendalltau(data[i], data[j])
        return (i, j, tau)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: compute_tau(*p), pairs)

    for i, j, tau in results:
        tau_matrix[i, j] = tau
        tau_matrix[j, i] = tau

    """ IL BRO HA CAPPATO! STA BARANDO SUI RISULTATI
    # Step 2: Convert Kendall tau to Pearson correlation
    corr_matrix = np.sin((np.pi / 2) * tau_matrix)

    # Optional PSD cleaning
    if clean_psd:
        corr_matrix = nearest_psd(corr_matrix)

    # Step 3: Build covariance matrix
    stds = np.std(data, axis=1, ddof=1)
    D = np.diag(stds)
    cov_matrix =D @ corr_matrix @ D
    
    return cov_matrix
    """
    

    return tau_matrix
    

   


def nearest_psd(matrix):
    """Project matrix to the nearest positive semidefinite matrix using eigenvalue clipping."""
    eigvals, eigvecs = eigh(matrix)
    eigvals[eigvals < 0] = 0  # clip negative eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def mutual_info_matrix_parallel(data, max_workers=None):
    """
    Versione parallela della matrice di mutua informazione.

    Parameters:
    - data: np.ndarray (N, T)
    - max_workers: numero massimo di thread

    Returns:
    - mi_matrix: np.ndarray (N, N)
    """
    data_t = data.T  # shape: (T, N)
    N = data_t.shape[1]
    mi_matrix = np.zeros((N, N))

    def compute_mi(i, j):
        if i == j:
            mi = mutual_info_regression(data_t[:, i].reshape(-1, 1), data_t[:, i])[0]
        else:
            mi = mutual_info_regression(data_t[:, i].reshape(-1, 1), data_t[:, j])[0]
        return (i, j, mi)

    pairs = [(i, j) for i in range(N) for j in range(i, N)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: compute_mi(*p), pairs)

    for i, j, mi in results:
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi

    return mi_matrix

import time

def Compute_Performances_Rolling(
    X_train_3D, X_test_3D, Oracle_Train_3D, Oracle_Test_3D,
    pathfilename_temp=None, OUTPUT=None, Compute_MI=False
):
    n_windows = X_train_3D.shape[0]
    strategies = ["min_var ", "omn     ", "mean_rev", "rnd     "]
    methods_list = ["Sample ", "Rie    ", "IW     ", "Clipped", 
                    "Shrunk ", "Kendall", "TMFG   ", "TMFG_MI"]
    n_methods = len(methods_list)
    stepTotali = n_windows - 1

    # Inizializza dizionario per varianze e tempi
    rolling_performance_dict = {
        (strategy, method): {"risks": [], "time": []}
        for strategy in strategies for method in methods_list
    }

    for i in range(stepTotali):
        BarraCaricamento(stepTotali, i)
        X_train = X_train_3D[i]
        X_test = X_test_3D[i]
        Oracle_train = Oracle_Train_3D[i]
        Oracle_test = Oracle_Test_3D[i]

        X_train_std, std_daily, std_stocks = standardize_returns(X_train)
        X_test_std, _, _ = standardize_returns(X_test)
        Oracle_test_std, _, _ = standardize_returns(Oracle_test)

        # === Inizio misura tempi ===
        start = time.time()
        E_sample = np.cov(X_train)
        t_sample = time.time() - start

        start = time.time()
        E_rie = rmt.optimalShrinkage(X_train, return_covariance=True, method="rie")
        t_rie = time.time() - start

        start = time.time()
        E_iw = rmt.optimalShrinkage(X_train, return_covariance=True, method="iw")
        t_iw = time.time() - start

        start = time.time()
        E_Clipped = rmt.clipped(X_train, alpha=0.0, return_covariance=True)
        t_clipped = time.time() - start

        start = time.time()
        E_shrunk = shrunk_covariance(E_sample, shrinkage=0.1)
        t_shrunk = time.time() - start

        start = time.time()
        E_Kendall = kendall_tau_matrix_parallel(X_train, clean_psd=True)
        t_kendall = time.time() - start

        model = tmfg.TMFG()
        corr = np.square(np.corrcoef(X_train, rowvar=True))

        start = time.time()
        _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_sample, output="logo")
        t_tmfg = time.time() - start
        E_TMFG = np.linalg.inv(J_TMFG)

        start = time.time()
        if Compute_MI:
            MI = mutual_info_matrix_parallel(X_train)
            np.save("MI_Matrix.npy", MI)
        else:
            MI = np.load("MI_Matrix.npy")

        _, _, J_TMFG_MI = model.fit_transform(weights=MI, cov=E_sample, output="logo")
        t_tmfg_mi = time.time() - start
        E_TMFG_MI = np.linalg.inv(J_TMFG_MI)
        # === Fine misura tempi ===

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
        if not is_positive_definite(E_Kendall):
            print("Kendall covariance matrix not positive definite!\n It may happen!")
        if not is_positive_definite(J_TMFG):
            raise ValueError("Precision TMFG matrix not positive definite!")
        if not is_positive_definite(E_TMFG):
            raise ValueError("TMFG covariance matrix not positive definite!")
        if not is_positive_definite(J_TMFG_MI):
            raise ValueError("Precision TMFG_MI matrix not positive definite!")
        if not is_positive_definite(E_TMFG_MI):
            raise ValueError("TMFG_MI covariance matrix not positive definite!")

        Sigma_methods = {
            "Sample ": (E_sample, t_sample),
            "Rie    ": (E_rie, t_rie),
            "IW     ": (E_iw, t_iw),
            "Clipped": (E_Clipped, t_clipped),
            "Shrunk ": (E_shrunk, t_shrunk),
            "Kendall": (E_Kendall, t_kendall),
            "TMFG   ": ((E_sample, J_TMFG), t_tmfg),
            "TMFG_MI": ((E_sample, J_TMFG_MI), t_tmfg_mi),
        }

        Optimal_Weights_dict = {}
        for strategy in strategies:
            for method, (Sigma, elapsed_time) in Sigma_methods.items():
                if method in ["TMFG   ", "TMFG_MI"]:
                    E_cov, J_prec = Sigma
                    _, w, _ = portfolio_statistics(X_train, E_cov, Oracle_train, Oracle_test, std_daily, std_stocks, strategy=strategy, J_Precision=J_prec)
                else:
                    _, w, _ = portfolio_statistics(X_train, Sigma, Oracle_train , Oracle_test, std_daily=None, std_stocks=None, strategy=strategy)

                Optimal_Weights_dict[(strategy, method)] = (w, elapsed_time)

        for (strategy, method), (w, elapsed_time) in Optimal_Weights_dict.items():
            risk = Risk_Out(X_test_std, w)
            rolling_performance_dict[(strategy, method)]["risks"].append(risk)
            rolling_performance_dict[(strategy, method)]["time"].append(elapsed_time)

        if pathfilename_temp is not None:
            save_performance_dict(rolling_performance_dict, filename=pathfilename_temp)

    print("\nSUMMARY STATISTICS OVER ROLLING WINDOWS\n")
    print("STRATEGY |  METHOD |  MEAN VARIANCE       |  CV%")
    print("---------------------------------------------------")
    index = 0
    for key, values in rolling_performance_dict.items():
        risks = np.array(values["risks"])
        mean_val = np.mean(risks)
        std_val = np.std(risks, ddof=1)
        cv = 100 * std_val / np.abs(mean_val) if mean_val != 0 else 0
        strategy, method = key
        print(f"{strategy} | {method} | {mean_val:.2e} +/- {std_val:.1e} | {cv:.1f}")
            
        
        index += 1
        if (index % n_methods) == 0:
            print("---------------------------------------------------")
    print("---------------------------------------------------")

    save_performance_dict(rolling_performance_dict, filename="risultati_rolling_def.pkl")

    if OUTPUT is not None:
        Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT)

    return rolling_performance_dict
def save_performance_dict(performance_dict, filename="rolling_performance.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(performance_dict, f)

def load_and_summarize_performance(filename="rolling_performance.pkl", OUTPUT=None):
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot
    """
    import pickle
    import numpy as np

    with open(filename, "rb") as f:
        rolling_performance_dict = pickle.load(f)

    index = 0
    n_methods = 8
    print("\nSUMMARY STATISTICS OVER ROLLING WINDOWS\n")
    print("STRATEGY |  METHOD |  MEAN VARIANCE       |  CV%")
    print("---------------------------------------------------")
    for key, values in rolling_performance_dict.items():
        risks = np.array(values["risks"])
        mean_val = np.mean(risks)
        std_val = np.std(risks, ddof=1)
        cv = 100 * std_val / np.abs(mean_val) if mean_val != 0 else 0
        strategy, method = key
        print(f"{strategy} | {method} | {mean_val:.2e} +/- {std_val:.1e} | {cv:.1f}")
        index += 1
        if (index % n_methods) == 0:
            print("---------------------------------------------------")
    #print("---------------------------------------------------")

    if OUTPUT is not None:
        Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT)

def BarraCaricamento(stepTotali, step):
    terminal_size = shutil.get_terminal_size()
    larghezza_terminale = terminal_size.columns

    percentuale = step * 100 / stepTotali
    lunghezza_barra =larghezza_terminale-10
    progress = int(lunghezza_barra * step / stepTotali)  
    barra = '=' * progress + ' ' * (lunghezza_barra - progress) 
    
    print(f"\r[{barra}] {percentuale:.2f}%")





if __name__ == "__main__":
    pass
