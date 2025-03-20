import numpy as np
import matplotlib.pyplot as plt
#import pyRMT as rmt
#import yfinance as yf
import pandas as pd
import time
import os
import argparse
import traceback
from scipy.linalg import logm, inv


def load_stock_data(file_name="returns_data_1060.csv"):
    
    df = pd.read_csv(file_name, index_col=0)  # index_col=0 usa la prima colonna come indice (es. date)

    #metto a 0 i NaN
    df.fillna(0, inplace=True)

    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data=data.T
    X_train, X_test = data[:,:1000], data[:,1000:]
     
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
    X = (R - np.mean(R, axis=1).reshape(N, 1)) / np.std(R, axis=1).reshape(N, 1)
    X= X/ np.std(X, axis=0).reshape(1, T)
 
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

def omniscient_portfolio_vector(O_train ):
    """
    Compute the predictions vector  g for the omniscient portfolio.

    Parameters:
    O_train (numpy.ndarray): Matrix of oracle returns with T_out=60. (google colab for dataset creation)

   
    Returns:
    g (numpy.ndarray): Predictions vector for the omniscient portfolio.
    """
    
    N, T = O_train.shape
    g = np.sqrt(N)*O_train[:,0] / np.std(O_train, axis=1)
    
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
    g = -np.sqrt(N)*X_train[:,0]
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
    vec=vec/np.linalg.norm(vec)
    g= np.sqrt(N)*vec
    return g



   
def main():

    # define parameters
    N=500
    T=1000
    T_out=60


    # Load the stock data
    X_train, X_test = load_stock_data()
   
    
    # Standardize the returns
    X_train = standardize_returns(X_train)
    X_test = standardize_returns(X_test)
    Oracle_train, Oracle_test = load_stock_data(file_name="oracle_returns_data_1060.csv")
    g_min = minimum_variance_portfolio_vector(X_train)
    g_omn = omniscient_portfolio_vector(Oracle_train)
    g_mr = mean_reversion_portfolio_vector(X_train)
    g_rnd = random_long_short_portfolio_vector(X_train)
    
    print(f"MIN DIM:{g_min.shape}")
    print(f"ORACLE DIM:{g_omn.shape}")
    print(f"MR DIM:{g_mr.shape}")
    print(f"RND DIM:{g_rnd.shape}")


if __name__ == "__main__":
    main()
    
   