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


def load_stock_data():
    
    df = pd.read_csv("log_returns_data_1060.csv", index_col=0)  # index_col=0 usa la prima colonna come indice (es. date)

    #metto a 0 i NaN
    df.fillna(0, inplace=True)

    # Converti il DataFrame in un array NumPy
    data = df.to_numpy()
    data=data.T
    X_train, X_test = data[:1000], data[1000:]
     
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

def omniscient_portfolio_vector(P, R, T_out=60):
    """
    Compute the predictions vector  g for the omniscient portfolio.

    Parameters:
    P (numpy.ndarray): Matrix of daily price.
    R (numpy.ndarray): Matrix of returns.
    T_out (int): Number of days for the out-of-sample period

   
    Returns:
    g (numpy.ndarray): Predictions vector for the omniscient portfolio.
    """
    
    N, T = P.shape
    g = np.mean(X, axis=1)
    
    return g