import numpy as np
import numpy as np
import financial_test as FCA
import pyRMT as rmt

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
    alphas = np.arange(0, 1.0, 0.05)
    alphas = np.arange(1.0, 0, -0.05)
    print("Alphas:", alphas)

    performances = []
    shrinked_matrices = []
    weights = []

    for alpha in alphas:
        shrinked = (1 - alpha) * Sigma + alpha * target
        shrinked_matrices.append(shrinked)

        if method in ["TMFG___", "TMFG_MI"]:
            E_cov, J_prec = Sigma
            _, w = FCA.portfolio_statistics(X_train, E_cov, method, Oracle_train_std, X_train_std, std_daily=None, std_stocks=None, strategy=strategy, J_Precision=J_prec)
        else:
            _, w = FCA.portfolio_statistics(X_train, Sigma, method, Oracle_train_std , X_train_std,  std_daily=None, std_stocks=None, strategy=strategy)
        weights.append(w)

        # Calcolo del rischio out-of-sample
        validation_data, _, _ = FCA.standardize_returns(X_val)
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


