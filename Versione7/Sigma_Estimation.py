import numpy as np
import financial_test as FCA
import pyRMT as rmt
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
# Local paths
tmfg_core_path = os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
mfcf_path = os.path.expanduser("~/Desktop/UCL/CODE/MFCF")


# Cluster  paths
#tmfg_core_path = os.path.expanduser("~/CODE/Triangulated_Maximally_Filtered_Graph")
#mfcf_path = os.path.expanduser("~/CODE/MFCF")

sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)

import TMFG_core as tmfg

def Sample_Covariance(X, alpha=None, shrinkage_type=None):
    """
    Calcola la matrice di covarianza con possibilità di shrinkage.

    Parameters:
    - X (np.ndarray): matrice dei dati, shape (N, T), N variabili, T osservazioni.
    - alpha (float or None): coefficiente di shrinkage tra 0 e 1. Se None, niente shrinkage.
    - shrinkage_type (str or None): 'identity' o 'diagonal'. Se None, niente shrinkage.

    Returns:
    - cov_matrix (np.ndarray): matrice di covarianza (shrinkata se alpha e shrinkage_type sono specificati).
    """
    N, T = X.shape
    # Covarianza standard (T-1 al denominatore)
    sample_cov = np.cov(X)

    # Se nessuno shrinkage è richiesto
    if alpha is None or shrinkage_type is None:
        return sample_cov

    if not (0 <= alpha <= 1):
        raise ValueError("alpha deve essere compreso tra 0 e 1")

    if shrinkage_type == "identity":
        target = np.identity(N) * np.trace(sample_cov) / N

    elif shrinkage_type == "diagonal":
        target = np.diag(np.diag(sample_cov))

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Shrinked covariance
    shrinked_cov = alpha * target + (1 - alpha) * sample_cov
    return shrinked_cov

def RIE_Estimator(X, alpha=None, shrinkage_type=None):
    """
    Calcola la matrice di covarianza con possibilità di shrinkage.

    Parameters:
    - X (np.ndarray): matrice dei dati, shape (N, T), N variabili, T osservazioni.
    - alpha (float or None): coefficiente di shrinkage tra 0 e 1. Se None, niente shrinkage.
    - shrinkage_type (str or None): 'identity' o 'diagonal'. Se None, niente shrinkage.

    Returns:
    - cov_matrix (np.ndarray): matrice di covarianza (shrinkata se alpha e shrinkage_type sono specificati).
    """
    N, T = X.shape
    # Covarianza standard (T-1 al denominatore)
    Sigma = rmt.optimalShrinkage(X, return_covariance=True, method="rie")

    # Se nessuno shrinkage è richiesto
    if alpha is None or shrinkage_type is None:
        return Sigma

    if not (0 <= alpha <= 1):
        raise ValueError("alpha deve essere compreso tra 0 e 1")

    if shrinkage_type == "identity":
        target = np.identity(N) * np.trace(Sigma) / N

    elif shrinkage_type == "diagonal":
        target = np.diag(np.diag(Sigma))

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Shrinked covariance
    shrinked_cov = alpha * target + (1 - alpha) * Sigma
    return shrinked_cov

def RIE_IW_Estimator(X, alpha=None, shrinkage_type=None):
    """
    Calcola la matrice di covarianza con possibilità di shrinkage.

    Parameters:
    - X (np.ndarray): matrice dei dati, shape (N, T), N variabili, T osservazioni.
    - alpha (float or None): coefficiente di shrinkage tra 0 e 1. Se None, niente shrinkage.
    - shrinkage_type (str or None): 'identity' o 'diagonal'. Se None, niente shrinkage.

    Returns:
    - cov_matrix (np.ndarray): matrice di covarianza (shrinkata se alpha e shrinkage_type sono specificati).
    """
    N, T = X.shape
    # Covarianza standard (T-1 al denominatore)
    Sigma = rmt.optimalShrinkage(X, return_covariance=True, method="iw")
    # Se nessuno shrinkage è richiesto
    if alpha is None or shrinkage_type is None:
        return Sigma

    if not (0 <= alpha <= 1):
        raise ValueError("alpha deve essere compreso tra 0 e 1")

    if shrinkage_type == "identity":
        target = np.identity(N) * np.trace(Sigma) / N

    elif shrinkage_type == "diagonal":
        target = np.diag(np.diag(Sigma))

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Shrinked covariance
    shrinked_cov = alpha * target + (1 - alpha) * Sigma
    return shrinked_cov

def Clipped_Estimator(X, alpha=None, shrinkage_type=None):
    """
    Calcola la matrice di covarianza con possibilità di shrinkage.

    Parameters:
    - X (np.ndarray): matrice dei dati, shape (N, T), N variabili, T osservazioni.
    - alpha (float or None): coefficiente di shrinkage tra 0 e 1. Se None, niente shrinkage.
    - shrinkage_type (str or None): 'identity' o 'diagonal'. Se None, niente shrinkage.

    Returns:
    - cov_matrix (np.ndarray): matrice di covarianza (shrinkata se alpha e shrinkage_type sono specificati).
    """
    N, T = X.shape
    # Covarianza standard (T-1 al denominatore)
    Sigma = rmt.clipped(X, alpha=None, return_covariance=True)
    # Se nessuno shrinkage è richiesto
    if alpha is None or shrinkage_type is None:
        return Sigma

    if not (0 <= alpha <= 1):
        raise ValueError("alpha deve essere compreso tra 0 e 1")

    if shrinkage_type == "identity":
        target = np.identity(N) * np.trace(Sigma) / N

    elif shrinkage_type == "diagonal":
        target = np.diag(np.diag(Sigma))

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Shrinked covariance
    shrinked_cov = alpha * target + (1 - alpha) * Sigma
    return shrinked_cov


def Kendall_Estimator(X, alpha=None, shrinkage_type=None):
    """
    Calcola la matrice di covarianza con possibilità di shrinkage.

    Parameters:
    - X (np.ndarray): matrice dei dati, shape (N, T), N variabili, T osservazioni.
    - alpha (float or None): coefficiente di shrinkage tra 0 e 1. Se None, niente shrinkage.
    - shrinkage_type (str or None): 'identity' o 'diagonal'. Se None, niente shrinkage.

    Returns:
    - cov_matrix (np.ndarray): matrice di covarianza (shrinkata se alpha e shrinkage_type sono specificati).
    """
    N, T = X.shape
    # Covarianza standard (T-1 al denominatore)
    Sigma = FCA.kendall_tau_matrix_parallel(X)
    # Se nessuno shrinkage è richiesto
    if alpha is None or shrinkage_type is None:
        return Sigma

    if not (0 <= alpha <= 1):
        raise ValueError("alpha deve essere compreso tra 0 e 1")

    if shrinkage_type == "identity":
        target = np.identity(N) * np.trace(Sigma) / N

    elif shrinkage_type == "diagonal":
        target = np.diag(np.diag(Sigma))

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Shrinked covariance
    shrinked_cov = alpha * target + (1 - alpha) * Sigma
    return shrinked_cov

def Fast_TMFG(X_train):
    model = tmfg.TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=True))
    E_Sample_TMFG = np.cov(X_train)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    return E_TMFG, J_TMFG

def TMFG_Adjacency_Matrix(X_train):
    """
    Compute the adjacency matrix of the TMFG graph.
    """
    model = tmfg.TMFG()
    corr = np.square(np.corrcoef(X_train, rowvar=True))
    E_Sample_TMFG = np.cov(X_train)
    _, _, AD_Matrix = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="unweighted_sparse_W_matrix")
    
    return AD_Matrix

import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def TMFG_Network_Visualization(X_train, tickers_with_sectors_file="tickers_with_sectors.csv", output_dir=None):
    """
    Visualizza la rete TMFG con etichette dei ticker e colorazione per settore.

    Parametri:
        X_train: matrice (n_samples, n_assets)
        tickers_with_sectors_file: file CSV con colonne 'Ticker' e 'Sector'
        output_dir: directory per salvare l'immagine, se specificato
    """
    # Calcola la matrice di adiacenza
    AD = TMFG_Adjacency_Matrix(X_train)

    # Costruisci il grafo
    G = nx.from_numpy_array(AD)

    # Leggi tickers e settori
    df_tickers = pd.read_csv(tickers_with_sectors_file)
    if len(df_tickers) != G.number_of_nodes():
        raise ValueError(f"Mismatch tra ticker ({len(df_tickers)}) e nodi ({G.number_of_nodes()})")

    tickers = df_tickers["Ticker"].tolist()
    sectors = df_tickers["Sector"].tolist()

    # Mappa nodi -> ticker
    G = nx.relabel_nodes(G, {i: tickers[i] for i in range(len(tickers))})

    # Costruisci dizionario nodo → settore
    node_sector = {tickers[i]: sectors[i] for i in range(len(tickers))}

    # Mappa settore → colore
    unique_sectors = sorted(set(sectors))
    cmap = cm.get_cmap("tab20", len(unique_sectors))
    sector_color_map = {sector: cmap(i) for i, sector in enumerate(unique_sectors)}
    node_colors = [sector_color_map[node_sector[node]] for node in G.nodes]

    # Layout e dimensioni
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [100 + 10 * d for _, d in G.degree()]

    # Disegno
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="black", alpha=0.9)

    # Legenda
    for sector, color in sector_color_map.items():
        plt.scatter([], [], c=[color], label=sector)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Settori")

    plt.title("Network TMFG con nodi colorati per settore", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # Salva o mostra
    if output_dir:
        plt.savefig(f"{output_dir}/tmfg_network_by_sector.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()








def compute_best_shrinkage_covariance(
    Sigma, X_train, X_val, Oracle_train_std, X_train_std,
    shrinkage_type="identity", method="Rie____", strategy="min_var_",  
):
    """
    Trova la miglior matrice di covarianza shrinkata secondo la funzione Risk_out, 
    usando validazione su X_val.

    Parameters:
    - sample_cov (np.ndarray): matrice di covarianza originale (N x N)
    - X_train (np.ndarray): matrice dei dati originali (N x T)
    - X_val (np.ndarray): matrice di validazione (N x T_val)
    - shrinkage_type (str): 'identity' o 'diagonal'

    Returns:
    - best_cov (np.ndarray): matrice shrinkata ottimale secondo la validazione
    """

    N = Sigma.shape[0]
    row_mean, std_daily, std_stocks = FCA.standardize_parameters(X_train)  #RIGA NUOVA

    # Definizione matrice target
    if shrinkage_type == "identity":
        trace_avg = np.trace(Sigma) / N
        target = np.identity(N) * trace_avg

    elif shrinkage_type == "diagonal":
        variances = np.var(X_train, axis=1, ddof=1)
        target = np.diag(variances)

    else:
        raise ValueError("shrinkage_type deve essere 'identity' o 'diagonal'")

    # Range di alpha da testare
    alphas = np.arange(0., 1.05, 0.05)
    #alphas = np.arange(1.0, 0, -0.05)
    print("Alphas:", alphas)

    performances = []
    shrinked_matrices = []
    weights = []

    for alpha in alphas:
        shrinked = (1 - alpha) * Sigma + alpha * target
        shrinked_matrices.append(shrinked)

        if method in ["TMFG___SI", "TMFG___SD"]:
            # Calcolo TMFG
            E_TMFG, J_TMFG = Fast_TMFG(X_train)
            _, w = FCA.portfolio_statistics(
                X_train,
                shrinked,
                method,
                Oracle_train_std,
                X_train_std,
                std_daily=None,
                std_stocks=None,
                strategy=strategy,
                J_Precision= J_TMFG
            )

        else:
            _, w = FCA.portfolio_statistics(
                X_train,
                shrinked,
                method,
                Oracle_train_std,
                X_train_std,
                std_daily=None,
                std_stocks=None,
                strategy=strategy
            )
        weights.append(w)

        # Calcolo rischio out-of-sample
        #validation_data, _, _ = FCA.standardize_returns(X_val) # STANDARDIZZAZIONE VECCHIA
        validation_data=FCA.normalize_returns(X_val) # STANDARDIZZAZIONE NUOVA

        #validation_data = (X_val - row_mean) / (std_stocks) #RIGA NUOVA 
        #validation_data = X_val # RIGA NUOVA
        risk = FCA.Risk_Out(validation_data, w, method, strategy)
        performances.append(risk)

    # Selezione migliore
    best_idx = np.argmin(performances)
    best_cov = shrinked_matrices[best_idx]
    best_weights = weights[best_idx]
    best_alpha = alphas[best_idx]
    print("Best alpha:", best_alpha)
    print("\n\n")

    return best_cov, best_weights


