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


def create_synthetic_dataset(N, T, C):
    """
    Crea un dataset sintetico di dimensione TxN con una matrice di covarianza C nota a priori.
    
    Parametri:
    N (int): Numero di variabili.
    T (int): Numero di osservazioni.
    C (numpy.ndarray): Matrice di covarianza NxN.
    
    Ritorna:
    numpy.ndarray: Dataset sintetico di dimensione NxT.
    """
    mean = np.zeros(N)
    dataset = np.random.multivariate_normal(mean, C, T).T
    return dataset

def l2_norm_difference(matrix1, matrix2):
    """
    Calcola la norma L2 della differenza tra due matrici.

    Parametri:
    matrix1 (numpy.ndarray): Prima matrice.
    matrix2 (numpy.ndarray): Seconda matrice.

    Ritorna:
    float: Norma L2 della differenza tra le due matrici.
    """
    return np.linalg.norm(matrix1 - matrix2, ord='fro')

def evaluate_estimators(N, T, C, M):
    """
    Valuta le performance dei diversi metodi di shrinkage ripetendo M volte la stima.
    
    Parametri:
    N (int): Numero di variabili.
    T (int): Numero di osservazioni.
    C (numpy.ndarray): Matrice di covarianza NxN.
    M (int): Numero di ripetizioni.
    
    Genera un unico istogramma comparativo per tutti i metodi.
    """
    errors = {"sample": [], "rie": [], "iw": [], "kernel": []}
    
    for _ in range(M):
        dataset = create_synthetic_dataset(N, T, C)
        E0 = np.cov(dataset)
        errors["sample"].append(l2_norm_difference(E0, C))
        
        E1 = rmt.optimalShrinkage(dataset, return_covariance=False, method='rie')
        errors["rie"].append(l2_norm_difference(E1, C))
        
        E2 = rmt.optimalShrinkage(dataset, return_covariance=False, method='iw')
        errors["iw"].append(l2_norm_difference(E2, C))
        
        E3 = rmt.optimalShrinkage(dataset, return_covariance=False, method='kernel')
        errors["kernel"].append(l2_norm_difference(E3, C))
    
    # Plot unico con tutti gli istogrammi
    plt.figure(figsize=(10, 6))
    
    methods = {"sample": "blue", "rie": "red", "iw": "green", "kernel": "purple"}
    
    for method, color in methods.items():
        plt.hist(errors[method], bins=20, histtype='step', linewidth=2, edgecolor=color, label=method)
    
    plt.title("Distribuzione degli errori per i diversi metodi")
    plt.xlabel("L2 Norm Difference")
    plt.ylabel("Frequenza")
    plt.legend()
    plt.show()

def plot_eigenvalues(E0, E1, E2, E3, E_Clipped, Oracle=None):
    """
    Plotta gli autovalori delle matrici ripulite rispetto agli autovalori campionari.
    """
    eig_E0 = np.linalg.eigvalsh(E0)
    #print(f"Autovalori campionari: {eig_E0}")
   # eig_E0=np.sort(eig_E0)  # Ordina e inverte l'ordin
    eig_E1 = np.linalg.eigvalsh(E1)
    eig_E2 = np.linalg.eigvalsh(E2)
    #print(f"Autovalori ripuliti IW: {eig_E2}")
    #eig_E2=np.sort(eig_E2) # Ordina e inverte l'ordin
    #print(f"Autovalori ripuliti IW ordinati: {eig_E2}")
    eig_E3 = np.linalg.eigvalsh(E3)
    eig_clip = np.linalg.eigvalsh(E_Clipped)


    
        
    
    plt.figure(figsize=(8, 6))
    plt.plot(eig_E0, eig_E1, 'o', label='RIE', alpha=0.7)
    plt.plot(eig_E0, eig_E2, 's', label='IW', alpha=0.7)
    plt.plot(eig_E0, eig_E3, 'd', label='Kernel', alpha=0.7)
    plt.plot(eig_E0, eig_clip, 'x', label='Clipped', alpha=0.7)
    
    if Oracle is not None:
        eig_Oracle = np.linalg.eigvalsh(Oracle)
        plt.plot(eig_E0, eig_Oracle, 'o', label='Oracle', alpha=0.7)
    
    plt.xlabel("Autovalori campionari")
    plt.ylabel("Autovalori ripuliti")
    plt.legend()
    plt.title("Autovalori ripuliti vs. Autovalori campionari")
    plt.show()

def metropolis_hastings_covariance(N, k, num_samples=1000, burn_in=500):
    # [Funzione invariata]
    C = np.eye(N) + 0.1 * np.random.randn(N, N)
    C = (C + C.T) / 2
    C = C @ C.T

    for _ in range(num_samples + burn_in):
        perturbation = 0.1 * np.random.randn(N, N)
        perturbation = (perturbation + perturbation.T) / 2
        C_new = C + perturbation
        C_new = (C_new + C_new.T) / 2

        if np.all(np.linalg.eigvals(C_new) > 0):
            log_prob_old = -N * np.trace((k + 1) * logm(C) + k * inv(C))
            log_prob_new = -N * np.trace((k + 1) * logm(C_new) + k * inv(C_new))
            
            if np.random.rand() < np.exp(log_prob_new - log_prob_old):
                C = C_new
    
    return C

def load_stock_data():
    
    df = pd.read_csv("log_returns_data_1060.csv", index_col=0)  # index_col=0 usa la prima colonna come indice (es. date)

    #metto a 0 i NaN
    df.fillna(0, inplace=True)
    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data=data.T
    X_train, X_test = data[:1000], data[1000:]
    if  np.isnan(X_train).any()==True:
        print("La matrice contiene NaN")
        contains_nan = True
        nan_positions = np.argwhere(np.isnan(X_train))
        print("Posizioni dei NaN:", nan_positions)

        # Sostituisci i NaN con  valore piccolo
        for i, j in nan_positions:
            X_train[i, j] =1e-4
     
    return X_train, X_test

def main(method):
    N = 500
    T = 1000
    M = 100
    
    if method == "isotropic":
        C = np.eye(N)
        dataset = create_synthetic_dataset(N, T, C)

        #oracole estimation
        q=N/T
        k=1
        alpha=1/(1+2*q*k)
        Oracle = alpha*np.cov(dataset) + (1-alpha)*np.eye(N)
       

    
    elif method == "inverse_wishart":
        k = 10
        C = metropolis_hastings_covariance(N, k)
        dataset = create_synthetic_dataset(N, T, C)
    
    elif method == "financial":
        dataset, dataset_test  = load_stock_data()
        print(f"Dimensioni corrette dataset finanziario: {dataset.shape}")
    
    else:
        raise ValueError("Metodo non valido. Scegli tra 'isotropic', 'inverse_wishart' o 'financial'.")
    
    E0 = np.cov(dataset)
    E1 = rmt.optimalShrinkage(dataset, return_covariance=True, method='rie')
    E2 = rmt.optimalShrinkage(dataset, return_covariance=True, method='iw')
    E3 = rmt.optimalShrinkage(dataset, return_covariance=True, method='kernel')
    E3=np.zeros((N,N))
    E_Clipped=rmt.clipped(dataset, alpha=0.2, return_covariance=True)
    
    
    plot_eigenvalues(E0, E1, E2,E3, E_Clipped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi delle matrici di covarianza")
    parser.add_argument("--method", 
                       type=str, 
                       choices=["isotropic", "inverse_wishart", "financial"],
                       required=True,
                       help="Metodo da utilizzare: isotropic, inverse_wishart o financial")
    
    args = parser.parse_args()
    
    try:
        main(args.method)
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        traceback.print_exc()  # Stampa il traceback completo con la linea di errore