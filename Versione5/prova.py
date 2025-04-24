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

# Set path for local modules
tmfg_core_path = os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
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

# Argument parser per leggere i parametri da riga di comando
parser = argparse.ArgumentParser(description="Esegui performance rolling con len_rolling variabile.")
parser.add_argument("--len_rolling", type=int, default=100, help="Dimensione della finestra rolling (default: 100)")
args = parser.parse_args()

# Imposta il seed
np.random.seed(27029)

# Parametri rolling
len_rolling = args.len_rolling

# Crea la directory di output se non esiste
output_dir = f"Rolling_{len_rolling}"
os.makedirs(output_dir, exist_ok=True)

# Carica i dati rolling
X_train, X_test = FCA.load_stock_data_rolling(file_name="returns_data_1060.csv", len_rolling=len_rolling)
Oracle_train, Oracle_test = FCA.load_stock_data_rolling(file_name="oracle_returns_data_1060.csv", len_rolling=len_rolling)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Calcola le performance rolling
save_path_temp = os.path.join(output_dir, "risultati_rolling_temp.pkl")
rolling_performance_dict = FCA.Compute_Performances_Rolling(
    X_train, X_test, Oracle_train, Oracle_test, 
    OUTPUT=None, 
    pathfilename_temp=save_path_temp, 
    Compute_MI=True
)

# Salva i risultati nella directory corretta
save_path = os.path.join(output_dir, "risultati_rolling.pkl")
FCA.save_performance_dict(rolling_performance_dict, filename=save_path)