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

# Rileva l'ambiente: default = "local"
env = os.environ.get("ENV", "local")

if env == "local":
    tmfg_core_path = os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    mfcf_path = os.path.expanduser("~/Desktop/UCL/CODE/MFCF")
elif env == "cluster":
    tmfg_core_path = os.path.expanduser("~/CODE/Triangulated_Maximally_Filtered_Graph")
    mfcf_path = os.path.expanduser("~/CODE/MFCF")
else:
    raise ValueError(f"[ERROR] ENV='{env}' non riconosciuto. Usa 'local' o 'cluster'.")

# Aggiungi i path al sys.path
sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path) 

import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table
import financial_test as FCA
import Sigma_Estimation as SE
import pickle


import pandas as pd
import numpy as np

def load_stock_data_rolling(
    file_name="returns_data_1060.csv", 
    train_size=400, 
    val_size=60,
    N_stocks=400, 
    T_out=60, 
    len_rolling=100,
    fixed_val_start=800
):
    """
    Carica i dati e genera triple rolling:
    - validation sempre fissata a fixed_val_start
    - test sempre fissato a fixed_val_start + val_size
    - train che termina in fixed_val_start e parte a ritroso di train_size
    """

    df = pd.read_csv(file_name, index_col=0)

    if df.isna().values.any():
        print(f"Total number of NaN values before interpolation: {df.isna().sum().sum()}")
        df.interpolate(method="linear", inplace=True)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
    if df.isna().values.any():
        print("Ancora NaN trovati dopo il riempimento, verranno sostituiti con 0.")
        df.fillna(0, inplace=True)

    data = df.to_numpy().T  # shape: (N_assets, T)
    data = data[:N_stocks, :]
    num_assets, num_timesteps = data.shape

    # Compute starting index for first training window
    train_start_min = fixed_val_start - train_size

    if train_start_min < 0:
        raise ValueError(f"Train size {train_size} troppo lungo rispetto a fixed_val_start={fixed_val_start}")

    train_windows = []
    val_windows = []
    test_windows = []

    start_idx = train_start_min
    while True:
        train_start = start_idx
        train_end = train_start + train_size
        val_start = fixed_val_start + (start_idx - train_start_min)
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + T_out

        if test_end > num_timesteps:
            break  # stop rolling quando non ci sono più dati a sufficienza

        train_block = data[:, train_start:train_end]
        val_block = data[:, val_start:val_end]
        test_block = data[:, test_start:test_end]

        train_windows.append(train_block)
        val_windows.append(val_block)
        test_windows.append(test_block)

        start_idx += len_rolling

    if len(train_windows) == 0:
        raise ValueError("Nessuna finestra valida trovata: controlla la lunghezza dei dati o i parametri.")

    train_data = np.stack(train_windows)
    val_data = np.stack(val_windows)
    test_data = np.stack(test_windows)

    return train_data, val_data, test_data

def standardize_returns(R):
    """
    Standardize the returns matrix R.

    Parameters:
    R (numpy.ndarray): Matrix of returns.

    row_mean: temporal mean for each stock (N x 1)
    std_daily: daily standard deviation for each stock (N x 1)
    std_stocks: standard deviation for each stocks (1 x T) -> remove eteroschedasticity of data
    

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
    std_stocks = np.std(X, axis=0, keepdims=True, ddof=0)   
    std_stocks[std_stocks < 1e-10] = 1  # Evita divisioni per zero
    #X = (X) / std_stocks # RIGA NUOVA

    # Standardizzazione sulle righe (asse 1)
    row_mean = np.mean(X, axis=1, keepdims=True)
    std_daily = np.std(X, axis=1, keepdims=True, ddof=0) *np.sqrt(T)
    std_daily[std_daily < 1e-10] = 1  # Evita divisioni per zero
    X = (X - row_mean) / std_daily
    return X, std_daily, std_stocks

import numpy as np

def normalize_returns(X):
    """
    Normalizza la matrice X (N x T):
    - rimuove la media temporale per ogni stock
    - normalizza per la volatilità giornaliera stimata cross-section

    Parameters:
    -----------
    X : np.ndarray
        Matrice di shape (N, T) con i rendimenti

    Returns:
    --------
    X_norm : np.ndarray
        Matrice normalizzata
    """
    # Rimuovo la media temporale per ogni stock
    X_demeaned = X - np.mean(X, axis=1, keepdims=True)

    # Calcolo la volatilità giornaliera stimata per ogni colonna (istante temporale)
    sigma_hat = np.sqrt(np.sum(X_demeaned**2, axis=0, keepdims=True))

    # Evitiamo eventuali divisioni per zero
    sigma_hat[sigma_hat == 0] = 1.0

    # Normalizzo
    X_norm = X_demeaned / sigma_hat

    return X_norm

def standardize_parameters(R):
    """ 
        Return the parameters to standardize: row_mean, std_daily, std_stocks.
        row_mean: temporal mean for each stock (N x 1)
        std_daily: daily standard deviation for each stock (N x 1)
        std_stocks: standard deviation for each stocks (1 x T) -> remove eteroschedasticity of data
    """

    N, T = R.shape
    X = R
    # Standardizzazione sulle colonne (asse 0)
    col_mean = np.mean(X, axis=0, keepdims=True)
    std_stocks = np.std(X, axis=0, keepdims=True, ddof=0)  
    std_stocks[std_stocks < 1e-10] = 1  # Evita divisioni per zero
    #X = (X) / std_stocks # RIGA NUOVA

    # Standardizzazione sulle righe (asse 1)
    row_mean = np.mean(X, axis=1, keepdims=True)
    std_daily = np.std(X, axis=1, keepdims=True, ddof=0)  *np.sqrt(T)
    std_daily[std_daily < 1e-10] = 1  # Evita divisioni per zero
    X = (X - row_mean) / std_daily

    return row_mean, std_daily, std_stocks


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

    #if var < 0:
    #    raise ValueError(f"{strategy}_{method}-Portfolio variance is negative!")

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

import matplotlib.pyplot as plt
import os
from collections import defaultdict

def Show_Outliers(variance_data, OUTPUT="Single_Boxplot", Save=False, plots_dir=None, len_rolling=100, Q=0.5, log_scale=False): 
    """
    Show the outliers of the variance data using boxplots.

    Parameters:
    - variance_data: Dictionary containing the variance data for each strategy and method.
       variance_data[(strategy, method)] = performances
    - OUTPUT: String indicating the type of boxplot to create.
        "Single_Boxplot" for a single boxplot per method.
        "Multiple_Boxplot" for grouped boxplots per strategy.
        "Shrinkage_Boxplot" to group by method base name (e.g., Sample_, Sample__SI, Sample__SD).
    - Save: Whether to save the plots or display them.
    - plots_dir: Where to save plots if Save=True.
    - len_rolling: Rolling window size for title/filename.
    - Q: A float, e.g., ratio N/T, used in title/filename.
    """

    if OUTPUT == "Single_Boxplot":
        for (strategy, methods), var in variance_data.items():
            plt.figure(figsize=(8, 6))
            plt.boxplot(var)
            if log_scale:
                plt.yscale('log')
            plt.title(f"Q={Q:.2f}, len_rolling={len_rolling}: Box Plot per {strategy}, {methods}")
            plt.ylabel("Variance")
            plt.xlabel("Method")
            plt.grid(True, linestyle="--", alpha=0.7)
            if Save and plots_dir:
                plt.savefig(f"{plots_dir}/Q_{Q:.2f}_{strategy}_{methods}_Rolling_{len_rolling}.png")
            else:
                plt.show()

    elif OUTPUT == "Multiple_Boxplot":
        grouped_data = {}
        for (strategy, methods), var in variance_data.items():
            if strategy not in grouped_data:
                grouped_data[strategy] = {}
            grouped_data[strategy][methods] = var

        for strategy, methods_data in grouped_data.items():
            plt.figure(figsize=(8, 6))
            data = list(methods_data.values())
            labels = list(methods_data.keys())
            plt.boxplot(data, labels=labels)
            if log_scale:
                plt.yscale('log')
            plt.title(f"Q={Q:.2f}, len_rolling={len_rolling}: Box Plot per {strategy}")
            plt.ylabel("Variance")
            plt.xlabel("Method")
            plt.xticks(rotation=30)
            plt.grid(True, linestyle="--", alpha=0.7)

            if Save and plots_dir:
                plt.savefig(f"{plots_dir}/Q_{Q:.2f}_{strategy}_Multiple_Rolling_{len_rolling}.png")
            else:
                plt.show()

    elif OUTPUT == "Shrinkage_Boxplot":
    # Per ogni strategia e metodo base, creiamo un gruppo con esattamente 3 slot: base, _SI, _SD
        for strategy in set(k[0] for k in variance_data.keys()):
            # Raggruppa i metodi che iniziano con lo stesso base_method
            base_methods = set()
            for (s, m) in variance_data.keys():
                if s != strategy:
                    continue
                if m.endswith("_SI") or m.endswith("_SD"):
                    base = m.rsplit("_", 1)[0]
                else:
                    base = m
                base_methods.add(base)

            for base in base_methods:
                variants = {
                    base: None,
                    f"{base}_SI": None,
                    f"{base}_SD": None
                }

                for variant in variants:
                    key = (strategy, variant)
                    if key in variance_data:
                        variants[variant] = variance_data[key]

                # Costruiamo il boxplot solo se almeno una variante ha dati
                if any(v is not None for v in variants.values()):
                    plt.figure(figsize=(8, 6))
                    data = [variants[base], variants[f"{base}_SI"], variants[f"{base}_SD"]]
                    labels = [base, f"{base}_SI", f"{base}_SD"]

                    # Rimuove i None ma mantiene l'ordine e le etichette corrispondenti
                    filtered_data_labels = [(d, l) for d, l in zip(data, labels) if d is not None]
                    if not filtered_data_labels:
                        continue
                    data, labels = zip(*filtered_data_labels)

                    plt.boxplot(data, labels=labels)
                    if log_scale:
                        plt.yscale('log')
                    plt.title(f"Q={Q:.2f}, len_rolling={len_rolling}: Shrinkage Box Plot - {strategy}, {base}")
                    plt.ylabel("Variance")
                    plt.xlabel("Method")
                    plt.xticks(rotation=30)
                    plt.grid(True, linestyle="--", alpha=0.7)

                    if Save and plots_dir:
                        filename = f"{plots_dir}/Q_{Q:.2f}_{strategy}_{base}_Shrinkage_Rolling_{len_rolling}.png"
                        plt.savefig(filename)
                    else:
                        plt.show()

    elif OUTPUT == "Comparison_Boxplot":
        for strategy in set(k[0] for k in variance_data.keys()):
            categories = {
                "Base": lambda m: not (m.endswith("_SI") or m.endswith("_SD")),
                "SI": lambda m: m.endswith("_SI"),
                "SD": lambda m: m.endswith("_SD"),
            }

            for category_name, method_filter in categories.items():
                # Filtra i metodi secondo la categoria attuale
                filtered_methods = [m for (s, m) in variance_data.keys() if s == strategy and method_filter(m)]
                if not filtered_methods:
                    continue

                data = []
                labels = []

                for method in filtered_methods:
                    key = (strategy, method)
                    if key in variance_data:
                        data.append(variance_data[key])
                        labels.append(method)

                if not data:
                    continue

                plt.figure(figsize=(10, 6))
                plt.boxplot(data, labels=labels)
                if log_scale:
                    plt.yscale('log')
                plt.title(f"Q={Q:.2f}, len_rolling={len_rolling}: {category_name} Methods Box Plot - {strategy}")
                plt.ylabel("$Portfolio Variance$")
                plt.xlabel("$Method")
                plt.xticks(rotation=30)
                plt.grid(True, linestyle="--", alpha=0.7)

                if Save and plots_dir:
                    filename = f"{plots_dir}/Q_{Q:.2f}_{strategy}_{category_name}_Comparison_Rolling_{len_rolling}.png"
                    plt.savefig(filename)
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
    pathfilename_temp=None, OUTPUT=None, Compute_MI=False, log_scale=False,
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
    rolling_weights_dict = { (strategy, method, step): [] for strategy in strategies for method in methods_list for step in range(stepTotali) }
    


    for strategy in strategies:
        for method in shrinkage_list:
            rolling_performance_dict[(strategy, method)] = []
            for step in range(stepTotali):
                rolling_weights_dict[(strategy, method, step)] = []
                
        
    for i in range(stepTotali):
        BarraCaricamento(stepTotali, i)

        X_train = X_train_3D[i]

        X_validation = X_validation_3D[i]
        X_test = X_test_3D[i]
        Oracle_train = Oracle_Train_3D[i]
        Oracle_validation = Oracle_validation_3D[i]
        Oracle_test = Oracle_Test_3D[i]
        row_mean, std_daily, std_stocks = standardize_parameters(X_train)  #RIGA NUOVA

        N, T = X_train.shape
        Q = N / T

        # Standardization VECCHIA MANIERA
        """ 
        X_train_std, std_daily, std_stocks = standardize_returns(X_train)
        X_test_std, _, _= standardize_returns(X_test)
        X_val_std, _, _= standardize_returns(X_validation) # DUBBIO
        Oracle_train_std, _ = standardize_returns_oracle(Oracle_train)
        
        
        """
        

        # Standardization NUOVA MANIERA
        X_train_std = normalize_returns(X_train)
        X_test_std= normalize_returns(X_test)
        Oracle_train_std, _ = standardize_returns_oracle(Oracle_train)

        X_train = X_train_std # DUBBIO

        #Oracle_test_std, _, _ = standardize_returns(Oracle_test)

        # Covariance estimators
        E_sample = SE.Sample_Covariance(X_train)
        E_rie = SE.RIE_Estimator(X_train)
        E_iw = SE.RIE_IW_Estimator(X_train,)
        E_Clipped = SE.Clipped_Estimator(X_train)
        E_Kendall = SE.Kendall_Estimator(X_train) # comment for Fast Experimets

        # TMFG e TMFG_MI
        E_TMFG, J_TMFG = SE.Fast_TMFG(X_train)

        Sigma_methods = {
            "Sample_": E_sample,
            "Rie____": E_rie,
            "IW_____": E_iw,
            "Clipped": E_Clipped,
            "Kendall": E_Kendall,  # comment for Fast Experimets
            "TMFG___": (E_sample, J_TMFG),
        }

        # In-sample: calcolo pesi
        Optimal_Weights_dict = {}
        for strategy in strategies:
            for method, Sigma in Sigma_methods.items():
                
                if method in ["TMFG___", "TMFG_MI"]:
                    E_cov, J_prec = Sigma
                    _, w = portfolio_statistics(X_train, E_cov, method, Oracle_train, Oracle_test, std_daily=None, std_stocks=None, strategy=strategy, J_Precision=J_prec)
                else:
                    _, w = portfolio_statistics(X_train, Sigma, method, Oracle_train , Oracle_test, std_daily=None, std_stocks=None, strategy=strategy)

                Optimal_Weights_dict[(strategy, method)] = w
                rolling_weights_dict[(strategy, method, i)].append(w)  # Salva i pesi per ogni rolling window

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
                        X_val=X_validation, # DUBBIO -> viene standardizzato nella funzione
                        Oracle_train_std=FCA.standardize_returns_oracle(Oracle_train),
                        X_train_std=FCA.standardize_returns_oracle(X_train),
                        shrinkage_type=target,
                        method=method_name,
                        strategy=strategy,
                    )

                    Optimal_Weights_dict[(strategy, method_name)] = w
                    rolling_weights_dict[(strategy, method_name, i)].append(w) 

        # Out-of-sample: calcolo rischio su X_test_std
        for (strategy, method), w in Optimal_Weights_dict.items():
            test_data = X_test_std
            #test_data = (X_test - row_mean) / (std_stocks) # RIGA NUOVA
            #test_data = X_test # RIGA NUOVA
            risk = Risk_Out(test_data, w, method, strategy)
            rolling_performance_dict[(strategy, method)].append(risk)
        if pathfilename_temp is not None:
            new_path = pathfilename_temp.replace(".pkl", "_weights.pkl")
            save_performance_dict(rolling_performance_dict, filename=pathfilename_temp)
            save_performance_dict(rolling_weights_dict, filename=new_path)
        
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
    #save_performance_dict(rolling_performance_dict, filename="risultati_rolling_def.pkl")
    #save_performance_dict(rolling_weights_dict, filename="weights_risultati_rolling_def.pkl")

    # Outlier plot
    if OUTPUT is not None:
        Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT, Q = Q, log_scale=log_scale)

    return rolling_performance_dict, rolling_weights_dict

def save_performance_dict(performance_dict, filename="rolling_performance.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(performance_dict, f)


def load_and_summarize_performance(filename="rolling_performance.pkl", 
                                   OUTPUT=None, 
                                   Save=False, 
                                   len_rolling=100, 
                                   Q = 0.5, 
                                   log_scale=False,
                                   output_dir=None):
    
    """
    OUTPUT Visualization of the outliers:
        -Single_Boxplot
        -Multiple_Boxplot
        -Shrinkage_Boxplot

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
            if output_dir:
                Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT, Save=True, plots_dir=output_dir, len_rolling=len_rolling, Q = Q, log_scale=log_scale)
            else:
                Show_Outliers(rolling_performance_dict, OUTPUT=OUTPUT, Save=True, plots_dir=plots_dir, len_rolling=len_rolling, Q = Q, log_scale=log_scale)
    

def BarraCaricamento(stepTotali, step):
    terminal_size = shutil.get_terminal_size()
    larghezza_terminale = terminal_size.columns

    percentuale = step * 100 / stepTotali
    lunghezza_barra =larghezza_terminale-10
    progress = int(lunghezza_barra * step / stepTotali)  
    barra = '=' * progress + ' ' * (lunghezza_barra - progress) 
    
    #print(f"\r[{barra}] {percentuale:.2f}%")
    print(f"\r[{barra}] {step}/ {stepTotali}")




from scipy.stats import entropy
from scipy.special import softmax

def entropy_absolute_weights(w, base=2):
    """
    Computes the entropy of a weight vector.
    w => softamx: p => S: entropy

    Parameters:
    w (numpy.ndarray): Weight vector.
    base (int): Base of the logarithm. Default is 2 for binary entropy.

    Returns:
    S (float): Entropy of the weight vector.
    """
    
    w = np.array(w)
    p = softmax(w)
   

    S = entropy(p, base=base)
   
    return S



def load_and_summarize_weights(filename="risultati_rolling_weights.pkl", 
                                   OUTPUT="Time_Boxplot", 
                                   Save=False, 
                                   len_rolling=100, 
                                   Q = 0.5, 
                                   log_scale=False,
                                   output_dir=None):
    
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
        rolling_weights_dict = pickle.load(f)

    # Raggruppa le entry per (strategy, method_name)
    grouped_data = defaultdict(lambda: {})
    # Converti in array 1D
    for (strategy, method_name, i), w in rolling_weights_dict.items():
            grouped_data[(strategy, method_name)][i] = np.array(w).flatten()

    if OUTPUT == "Entropy":
        strategy_filter = "min_var_"

        standard_methods = ["Sample_", "Rie____", "IW_____", "Clipped", "Kendall", "TMFG___"]
        suffix_SI = "_SI"
        suffix_SD = "_SD"

        categories = {
            "Standard": [],
            "Shrinkage_Identity": [],
            "Shrinkage_Diagonal": []
        }

        # Raggruppa per categoria
        for (strategy, method_name), time_dict in grouped_data.items():
            if strategy != strategy_filter:
                continue
            if method_name in standard_methods:
                categories["Standard"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SI):
                categories["Shrinkage_Identity"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SD):
                categories["Shrinkage_Diagonal"].append((method_name, time_dict))

        # Plot per ciascuna categoria
        for category, method_list in categories.items():
            plt.figure(figsize=(12, 6))

            for method_name, time_dict in method_list:
                sorted_indices = sorted(time_dict.keys())
                data = [time_dict[i] for i in sorted_indices]
                entropies = [entropy_absolute_weights(w) for w in data]

                plt.plot(sorted_indices, entropies, linestyle='-', marker='o', label=method_name)

            plt.title(f"Entropia dei pesi nel tempo - {category} - Strategy: {strategy_filter}")
            plt.xlabel("Temporal index (i)")
            plt.ylabel("Entropia (base 2)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    if OUTPUT == "Laverange_Portfolio":
        strategy_filter = "min_var_"

        standard_methods = ["Sample_", "Rie____", "IW_____", "Clipped", "Kendall", "TMFG___"]
        suffix_SI = "_SI"
        suffix_SD = "_SD"

        categories = {
            "Standard": [],
            "Shrinkage_Identity": [],
            "Shrinkage_Diagonal": []
        }

        # Raggruppa per categoria
        for (strategy, method_name), time_dict in grouped_data.items():
            if strategy != strategy_filter:
                continue
            if method_name in standard_methods:
                categories["Standard"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SI):
                categories["Shrinkage_Identity"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SD):
                categories["Shrinkage_Diagonal"].append((method_name, time_dict))

        def compute_leverage_series(data):
            return [np.sum(np.abs(weights)) for weights in data]

        # Plot per ciascuna categoria
        for category, method_list in categories.items():
            plt.figure(figsize=(12, 6))

            for method_name, time_dict in method_list:
                sorted_indices = sorted(time_dict.keys())
                data = [time_dict[i] for i in sorted_indices]

                if len(data) < 2:
                    continue  # turnover non definito

                leverange = compute_leverage_series(data)
                leverange_indices = sorted_indices

                plt.plot(leverange_indices, leverange, linestyle='-', marker='o', label=method_name)

            plt.title(f"Leverange Portfolio - {category} - Strategy: {strategy_filter}")
            plt.xlabel("Temporal index  (i)")
            plt.ylabel("Daily Leverange")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    if OUTPUT == "Participation_Ratio":
        strategy_filter = "min_var_"

        standard_methods = ["Sample_", "Rie____", "IW_____", "Clipped", "Kendall", "TMFG___"]
        suffix_SI = "_SI"
        suffix_SD = "_SD"

        categories = {
            "Standard": [],
            "Shrinkage_Identity": [],
            "Shrinkage_Diagonal": []
        }

        # Raggruppa per categoria
        for (strategy, method_name), time_dict in grouped_data.items():
            if strategy != strategy_filter:
                continue
            if method_name in standard_methods:
                categories["Standard"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SI):
                categories["Shrinkage_Identity"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SD):
                categories["Shrinkage_Diagonal"].append((method_name, time_dict))

        
        
        
        def compute_participation_ratio_series(data):
            return [1.0 / np.sum(w**4) for w in data]

        # Plot per ciascuna categoria
        for category, method_list in categories.items():
            plt.figure(figsize=(12, 6))

            for method_name, time_dict in method_list:
                sorted_indices = sorted(time_dict.keys())
                data = [time_dict[i] for i in sorted_indices]

                if len(data) < 2:
                    continue  # turnover non definito

                ParticipatioRatio = compute_participation_ratio_series(data)
                PR_indices = sorted_indices

                plt.plot(PR_indices, ParticipatioRatio, linestyle='-', marker='o', label=method_name)

            plt.title(f"Participation Ratio - {category} - Strategy: {strategy_filter}")
            plt.xlabel("Temporal index  (i)")
            plt.ylabel("Daily Leverange")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()




    if OUTPUT == "Daily_Turnover":
        strategy_filter = "min_var_"

        standard_methods = ["Sample_", "Rie____", "IW_____", "Clipped", "Kendall", "TMFG___"]
        suffix_SI = "_SI"
        suffix_SD = "_SD"

        categories = {
            "Standard": [],
            "Shrinkage_Identity": [],
            "Shrinkage_Diagonal": []
        }

        # Raggruppa per categoria
        for (strategy, method_name), time_dict in grouped_data.items():
            if strategy != strategy_filter:
                continue
            if method_name in standard_methods:
                categories["Standard"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SI):
                categories["Shrinkage_Identity"].append((method_name, time_dict))
            elif method_name.endswith(suffix_SD):
                categories["Shrinkage_Diagonal"].append((method_name, time_dict))

        def compute_turnover_series(data):
            return [np.sum(np.abs(data[i+1] - data[i])) for i in range(len(data)-1)]

        # Plot per ciascuna categoria
        for category, method_list in categories.items():
            plt.figure(figsize=(12, 6))

            for method_name, time_dict in method_list:
                sorted_indices = sorted(time_dict.keys())
                data = [time_dict[i] for i in sorted_indices]

                if len(data) < 2:
                    continue  # turnover non definito

                turnover = compute_turnover_series(data)
                turnover_indices = sorted_indices[1:]

                plt.plot(turnover_indices, turnover, linestyle='-', marker='o', label=method_name)

            plt.title(f"Daily Turnover - {category} - Strategy: {strategy_filter}")
            plt.xlabel("Temporal index (i)")
            plt.ylabel("Daily Turnover")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
            


    else:
          

        # Per ogni combinazione (strategy, method_name), produci il grafico
        for (strategy, method_name), time_dict in grouped_data.items():
            # Ordina gli step temporali
            sorted_indices = sorted(time_dict.keys())
            data = [time_dict[i] for i in sorted_indices]

            if OUTPUT == "Time_Boxplot":
                # Calcola le medie
                means = [np.mean(w) for w in data]
                medians = [np.median(w) for w in data]

                plt.figure(figsize=(12, 6))
                plt.boxplot(data, positions=sorted_indices, showfliers=False)
                plt.plot(sorted_indices, means, color='red', linestyle='-', marker='o', label='Media dei pesi')
                plt.plot(sorted_indices, medians, color='blue', linestyle='--', marker='x', label='Mediana dei pesi')

                plt.title(f"Distribuzione pesi - Strategy: {strategy}, Method: {method_name}")
                plt.xlabel("Temporal Index (i)")
                plt.ylabel("Weight")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        




if __name__ == "__main__":
    pass
