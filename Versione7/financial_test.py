import numpy as np
import matplotlib.pyplot as plt
import pyRMT as rmt
#import yfinance as yf
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

# Local paths
tmfg_core_path = os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
mfcf_path = os.path.expanduser("~/Desktop/UCL/CODE/MFCF")


# Cluster  paths
#tmfg_core_path = os.path.expanduser("~/CODE/Triangulated_Maximally_Filtered_Graph")
#mfcf_path = os.path.expanduser("~/CODE/MFCF")

sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)

import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table
import financial_test as FCA
import Sigma_Estimation as SE
import pickle


def load_stock_data_rolling(
    file_name="returns_data_1060.csv", 
    train_size=800, 
    val_size=60,
    N_stocks=400, 
    T_out=60, 
    len_rolling=100
):
    """
    Carica i dati delle azioni e genera triple rolling (training, validation, test) non sovrapposte.

    Parameters:
    - file_name: nome del file CSV contenente i rendimenti (shape: time x assets)
    - train_size: numero di timestep per ciascun blocco di training
    - val_size: numero di timestep per ciascun blocco di validazione
    - N_stocks: numero di asset da considerare
    - T_out: numero di timestep per ciascun blocco di test
    - len_rolling: lunghezza del rolling

    Returns:
    - train_data: array di shape (n_windows, N_stocks, train_size)
    - val_data: array di shape (n_windows, N_stocks, val_size)
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

    # Conversione in NumPy array e selezione asset
    data = df.to_numpy().T  # shape: (N_assets, T)
    data = data[:N_stocks, :]  # Seleziona i primi N_stocks

    num_assets, num_timesteps = data.shape

    # Generazione finestre rolling
    train_windows = []
    val_windows = []
    test_windows = []

    start_idx = 0
    total_window = train_size + val_size + T_out

    while start_idx + total_window <= num_timesteps:
        train_block = data[:, start_idx : start_idx + train_size]
        val_block = data[:, start_idx + train_size : start_idx + train_size + val_size]
        test_block = data[:, start_idx + train_size + val_size : start_idx + total_window]

        train_windows.append(train_block)
        val_windows.append(val_block)
        test_windows.append(test_block)

        start_idx += len_rolling  # Rolling di len_rolling timestep

    if len(train_windows) == 0:
        raise ValueError("Nessuna finestra valida trovata: controlla la lunghezza dei dati o i parametri train_size/val_size/T_out.")

    # Stack in array 3D
    train_data = np.stack(train_windows)  # shape: (n_windows, N_stocks, train_size)
    val_data = np.stack(val_windows)      # shape: (n_windows, N_stocks, val_size)
    test_data = np.stack(test_windows)    # shape: (n_windows, N_stocks, T_out)

    return train_data, val_data, test_data


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
    # Standardizzazione sulle colonne (asse 0)
    col_mean = np.mean(X, axis=0, keepdims=True)
    std_daily = np.std(X, axis=0, keepdims=True, ddof=0)   *np.sqrt(T)
    std_daily[std_daily < 1e-10] = 1  # Evita divisioni per zero
    X = (X) / std_daily

    # Standardizzazione sulle righe (asse 1)
    row_mean = np.mean(X, axis=1, keepdims=True)
    std_stocks = np.std(X, axis=1, keepdims=True, ddof=0)
    std_stocks[std_stocks < 1e-10] = 1  # Evita divisioni per zero
    X = (X - row_mean) / std_stocks

    return X, std_daily, std_stocks


def standardize_returns_oracle(R):
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
    # Standardizzazione sulle colonne (asse 0)
    #col_mean = np.mean(X, axis=0, keepdims=True)
    #std_daily = np.std(X, axis=0, keepdims=True, ddof=0)   *np.sqrt(T)
    #std_daily[std_daily < 1e-10] = 1  # Evita divisioni per zero
    #X = (X) / std_daily

    # Standardizzazione sulle righe (asse 1)
    row_mean = np.mean(X, axis=1, keepdims=True)
    std_stocks = np.std(X, axis=1, keepdims=True, ddof=0)
    std_stocks[std_stocks < 1e-10] = 1  # Evita divisioni per zero
    X = (X - row_mean) / std_stocks
   

    return X, std_stocks #, std_stocks



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
    O_train_std (numpy.ndarray): Matrix of stanadardized oracle returns with T_out=60. (google colab for dataset creation)


    Returns:
    g (numpy.ndarray): Predictions vector for the omniscient portfolio.
    """

    N, T = O_train_std.shape
    g = np.sqrt(N) * O_train_std[:, 0]
    return g


def mean_reversion_portfolio_vector(X_train_std):
    """
    Compute the predictions vector  g for the mean reversion portfolio.

    Parameters:
    X (numpy.ndarray): Matrix of standardized returns.
    T_out (int): Number of days for the prediction.

    Returns:
    g (numpy.ndarray): Predictions vector for the mean reversion portfolio.
    """

    N, T = X_train_std.shape
    g = -np.sqrt(N) * X_train_std[:, 0]
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

def optimal_weights_vecchia(Sigma, g, std_daily=None, std_stocks=None, J_Precision=None):
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


def optimal_weights(Sigma, g, std_daily=None, std_stocks=None, J_Precision=None):
    """
    Compute the optimal weights for the portfolio.

    Parameters:
    Sigma (numpy.ndarray): Covariance matrix of the returns.
    g (numpy.ndarray): Predictions vector for the portfolio.
    std_daily (numpy.ndarray): Daily volatilities (1 x T), unused here.
    std_stocks (numpy.ndarray): Stock volatilities (N x 1).
    J_Precision (numpy.ndarray or None): Optional precision matrix.

    Returns:
    w (numpy.ndarray): Optimal weights for the portfolio.
    """
    if J_Precision is not None:
        w = J_Precision @ g / (g @ J_Precision @ g)

    elif std_stocks is None:
        eigvals = np.linalg.eigvalsh(Sigma)
        #if np.min(np.abs(eigvals)) < 1e-10:
        #    raise ValueError("Autovalori troppo piccoli in Sigma. La matrice potrebbe essere quasi singolare.")
        
        Sigma_inv = np.linalg.inv(Sigma)
        w = Sigma_inv @ g / (g @ Sigma_inv @ g)

    else:
        Sigma_rescaled = (std_stocks @ std_stocks.T) * Sigma
        eigvals = np.linalg.eigvalsh(Sigma_rescaled)
        #if np.min(np.abs(eigvals)) < 1e-10:
        #    raise ValueError("Autovalori troppo piccoli in Sigma_rescaled. La matrice potrebbe essere quasi singolare.")

        Sigma_inv = np.linalg.inv(Sigma_rescaled)
        den = g @ Sigma_inv @ g
        w = ( Sigma_inv @ g ) / den


    return w


def variance_portfolio(Sigma, w, method, strategy, J_Precision=None):
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
        raise ValueError(f"{strategy}_{method}-Portfolio variance is negative!")

    return var


def portfolio_statistics(
    X_train, Sigma, method, Oracle_train_std, X_train_std, std_daily, std_stocks, strategy="min_var", J_Precision=None
):
    """
    Compute the statistics of the portfolio.

    Parameters:
    X_train (numpy.ndarray): Matrix of standardized returns.

    Sigma (numpy.ndarray): Covariance test matrix of the returns.
        Default is the identity matrix (Isotropic Case).

    method (str): Method for the covariance estimation.
        
    Oracle_train_std (numpy.ndarray): Matrix of standardized oracle returns.

    X_train_std (numpy.ndarray): Matrix of standardized returns.

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
    if strategy ==   "min_var_":
        g = minimum_variance_portfolio_vector(X_train)
    elif strategy == "omn_____":
        g = omniscient_portfolio_vector(Oracle_train_std)
    elif strategy == "mean_rev":
        g = mean_reversion_portfolio_vector(X_train_std)
    elif strategy == "rnd_____":
        g = random_long_short_portfolio_vector(X_train)

    g = check_norm(g)
    print(f"Norma del vettore g: {np.linalg.norm(g)}")

    # Optimal weights
    if J_Precision is not None:
        w = optimal_weights(Sigma, g, std_daily, std_stocks, J_Precision=J_Precision)
        var = variance_portfolio(Sigma, w, method, strategy)
    else:
        w = optimal_weights(Sigma, g, std_daily, std_stocks)
        var = variance_portfolio(Sigma, w, method, strategy)

    # print(f"Strategy: {strategy}, portfolio variance: {var}")

    return g, w


def is_positive_definite(matrix):
    # Calcola gli autovalori della matrice
    eigenvalues = np.linalg.eigvals(matrix)

    # Verifica se tutti gli autovalori sono positivi
    return np.all(eigenvalues > 0)


def Risk_Out(X, w, method, strategy):
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
    var = variance_portfolio(Sigma, w, method, strategy)
    print(f"{strategy} {method} Portfolio variance: {var}")

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


def Show_Outliers(variance_data, OUTPUT="Single_Boxplot", Save = False, plots_dir=None, len_rolling=100):
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
            if Save:
                plt.savefig(f"{plots_dir}/{strategy}_{methods}_Rolling_{len_rolling}.png")
            else:
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
            plt.xticks(rotation=30)  # Ruotiamo le etichette se sono lunghe
            plt.grid(True, linestyle="--", alpha=0.7)

            if Save:
                if plots_dir:
                    plt.savefig(f"{plots_dir}/{strategy}_Multiple_Rolling_{len_rolling}.png")

            else:
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

def kendall_tau_matrix_parallel(data, max_workers=None):
    """
    Versione parallela della matrice di Kendall Tau.

    Parameters:
    - data: np.ndarray (N, T)
    - max_workers: numero massimo di thread

    Returns:
    - tau_matrix: np.ndarray (N, N)
    """
    N = data.shape[0]
    tau_matrix = np.ones((N, N))
    stds = np.std(data, axis=1)

    def compute_tau(i, j):
        tau, _ = kendalltau(data[i], data[j])
        return (i, j, tau)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda p: compute_tau(*p), pairs)

    for i, j, tau in results:
        tau_matrix[i, j] = tau
        tau_matrix[j, i] = tau
    
    weight_matrix = np.outer(stds, stds)
    weighted_tau_matrix = tau_matrix * weight_matrix

    return weighted_tau_matrix

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


def Compute_Performances_Rolling(
    X_train_3D, X_validation_3D, X_test_3D, Oracle_Train_3D, Oracle_validation_3D, Oracle_Test_3D, 
    pathfilename_temp=None, OUTPUT=None, Compute_MI=False
):
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot
    Compute_MI:
        -True: calcola e salva la matrice della Mutua Informazione
        _FAlse: carica la matrice della mutua informazione
    """

    #n_windows = X_train_3D.shape[0]
    n_windows = Oracle_Train_3D.shape[0]
    strategies = ["min_var_",
                  "omn_____", 
                  "mean_rev", 
                  "rnd_____"]
    strategies = ["min_var_"] #debug

    methods_list =   ["Sample_", 
                      "Rie____", 
                      "IW_____", 
                      "Clipped", 
                      #"Shrunk_", 
                      "Kendall", # comment for Fast Experimets
                      "TMFG___",  ]
    shrinkage_list = ["Sample__SI", 
                      "Sample__SD",
                      "Rie_____SI",
                      "Rie_____SD",
                      "IW______SI",
                      "IW______SD",
                      "Clipped_SI",
                      "Clipped_SD",
                      "Kendall_SI", # comment for Fast Experimets
                      "Kendall_SD", # comment for Fast Experimets
                      "TMFG____SI",
                      "TMFG____SD",]

    n_methods = len(methods_list) + len(shrinkage_list)
    stepTotali=n_windows
    # Salva le performance per ogni coppia rolling (finestra train+test)
    rolling_performance_dict = { (strategy, method): [] for strategy in strategies for method in methods_list }
   
    for strategy in strategies:
        for method in shrinkage_list:
            rolling_performance_dict[(strategy, method)] = []
        
    for i in range(stepTotali):
        BarraCaricamento(stepTotali, i)

        X_train = X_train_3D[i]
        X_validation = X_validation_3D[i]
        X_test = X_test_3D[i]
        Oracle_train = Oracle_Train_3D[i]
        Oracle_validation = Oracle_validation_3D[i]
        Oracle_test = Oracle_Test_3D[i]

        # Standardization
        X_train_std, std_daily, std_stocks = standardize_returns(X_train)
        X_test_std, _, _= standardize_returns(X_test)
        Oracle_train_std, _ = standardize_returns_oracle(Oracle_train)



        #Oracle_test_std, _, _ = standardize_returns(Oracle_test)

        # Covariance estimators
        E_sample = SE.Sample_Covariance(X_train)
        E_rie = SE.RIE_Estimator(X_train)
        E_iw = SE.RIE_IW_Estimator(X_train,)
        E_Clipped = SE.Clipped_Estimator(X_train)
        #E_shrunk = shrunk_covariance(E_sample, shrinkage=0.1)
        E_Kendall = SE.Kendall_Estimator(X_train) # comment for Fast Experimets

        # TMFG e TMFG_MI
        E_TMFG, J_TMFG = SE.Fast_TMFG(X_train)

        Sigma_methods = {
            "Sample_": E_sample,
            "Rie____": E_rie,
            "IW_____": E_iw,
            "Clipped": E_Clipped,
            #"Shrunk_": E_shrunk,
            "Kendall": E_Kendall,  # comment for Fast Experimets
            "TMFG___": (E_sample, J_TMFG),
            #"TMFG_MI": (E_TMFG_MI, J_TMFG_MI),
        }

        # In-sample: calcolo pesi
        Optimal_Weights_dict = {}
        for strategy in strategies:
            for method, Sigma in Sigma_methods.items():
                
                if method in ["TMFG___", "TMFG_MI"]:
                    E_cov, J_prec = Sigma
                    _, w = portfolio_statistics(X_train, E_cov, method, Oracle_train, Oracle_test, std_daily, std_stocks, strategy=strategy, J_Precision=J_prec)
                else:
                    _, w = portfolio_statistics(X_train, Sigma, method, Oracle_train , Oracle_test, std_daily=None, std_stocks=None, strategy=strategy)

                Optimal_Weights_dict[(strategy, method)] = w

        # Add Shrinkage methods
        shrinkage_targets = ["identity", "diagonal"]
        base_estimators = {
            "Sample_": E_sample,
            "Rie____": E_rie,
            "IW_____": E_iw,
            "Clipped": E_Clipped,
            "Kendall": E_Kendall,  # comment for Fast Experimets
            "TMFG___": E_sample,
        }

        for strategy in strategies:
            for method_base, Sigma in base_estimators.items():
                for target in shrinkage_targets:
                    suffix = "SI" if target == "identity" else "SD"
                    method_name = method_base + "_" + suffix  # es: Sample__SI

                    _, w = SE.compute_best_shrinkage_covariance(
                        Sigma=Sigma,
                        X_train=X_train,
                        X_val=X_validation,
                        Oracle_train_std=FCA.standardize_returns_oracle(Oracle_train),
                        X_train_std=FCA.standardize_returns_oracle(X_train),
                        shrinkage_type=target,
                        method=method_name,
                        strategy=strategy,
                    )

                    Optimal_Weights_dict[(strategy, method_name)] = w


        # Out-of-sample: calcolo rischio su X_test_std
        for (strategy, method), w in Optimal_Weights_dict.items():
            test_data = X_test_std
            risk = Risk_Out(test_data, w, method, strategy)
            rolling_performance_dict[(strategy, method)].append(risk)
        if pathfilename_temp is not None:
            save_performance_dict(rolling_performance_dict, filename=pathfilename_temp)
        
    # Stampa finale delle statistiche sui rolling window
    index = 0
    print("\nSUMMARY STATISTICS OVER ROLLING WINDOWS\n")
    print("STRATEGY |  METHOD |  MEAN VARIANCE       |  CV%")
    print("---------------------------------------------------")
    for key, values in rolling_performance_dict.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = 100 * std_val / np.abs(mean_val) if mean_val != 0 else 0
        strategy, method = key
        print(f"{strategy} | {method} | {mean_val:.2e} +/- {std_val:.1e} | {cv:.1f}")
        index += 1
        if (index % n_methods) == 0:
            print("---------------------------------------------------")
    print("---------------------------------------------------\n\n")
    save_performance_dict(rolling_performance_dict, filename="risultati_rolling_def.pkl")


    # Outlier plot
    if OUTPUT is not None:
        Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT)

    return rolling_performance_dict

def save_performance_dict(performance_dict, filename="rolling_performance.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(performance_dict, f)


def load_and_summarize_performance(filename="rolling_performance.pkl", OUTPUT=None, Save=False, len_rolling=100):
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot

    Save: bool variable => saves plots
    """
    import pickle
    import numpy as np
    from pathlib import Path

    path = Path(filename)

    # Directory padre
    parent_dirs = path.parent         # Run_01/Rolling_10
    dir1 = parent_dirs.parent.name    # Run_01
    dir2 = parent_dirs.name           # Rolling_10

    # Nome del file
    file_name = path.name             # risultati_rolling_temp.pkl

    # Nuovo path: Run_01/Plots
    plots_dir = Path(dir1) / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)  # crea la directory se non esiste



    with open(filename, "rb") as f:
        rolling_performance_dict = pickle.load(f)

    index = 0
    n_methods = 7
    print("\nSUMMARY STATISTICS OVER ROLLING WINDOWS\n")
    print("STRATEGY |  METHOD |  MEAN VARIANCE       |  CV%")
    print("---------------------------------------------------")
    for key, values in rolling_performance_dict.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = 100 * std_val / np.abs(mean_val) if mean_val != 0 else 0
        strategy, method = key
        print(f"{strategy} | {method} | {mean_val:.2e} +/- {std_val:.1e} | {cv:.1f}")
        index += 1
        if (index % n_methods) == 0:
            print("---------------------------------------------------")
    print("---------------------------------------------------\n")

    if OUTPUT is not None:
        if Save:
            Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT, Save=True, plots_dir=plots_dir, len_rolling=len_rolling)
        else:
            Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT)

    

def BarraCaricamento(stepTotali, step):
    terminal_size = shutil.get_terminal_size()
    larghezza_terminale = terminal_size.columns

    percentuale = step * 100 / stepTotali
    lunghezza_barra =larghezza_terminale-10
    progress = int(lunghezza_barra * step / stepTotali)  
    barra = '=' * progress + ' ' * (lunghezza_barra - progress) 
    
    #print(f"\r[{barra}] {percentuale:.2f}%")
    print(f"\r[{barra}] {step}/ {stepTotali}")





if __name__ == "__main__":
    pass
