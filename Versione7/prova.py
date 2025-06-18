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
from sklearn.covariance import shrunk_covariance
from sklearn.feature_selection import mutual_info_regression

import sys

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

sys.path.append(tmfg_core_path)
sys.path.append(mfcf_path)


import TMFG_core as tmfg
import mfcf as mfcf
import gain_functions as gf
import gain_table
import financial_test as FCA

# Argument parser per leggere i parametri da riga di comando
parser = argparse.ArgumentParser(description="Esegui performance rolling con len_rolling variabile.")
parser.add_argument("--len_rolling", type=int, default=100, help="Dimensione della finestra rolling (default: 100)")
parser.add_argument("--N_stocks", type=int, default=400, help="Numero di azioni (default: 400)")
parser.add_argument("--training_size", type=int, default=800, help="Dimensione del training set (default: 500)")
parser.add_argument("--output", type=str, default=None, help="Single_Boxplot, Multiple_Boxplot, None (default: None)")
parser.add_argument("--cluster", type=bool, default=False, help="Aggiusta path per il cluster (default: False)")
args = parser.parse_args()

# Imposta il seed
np.random.seed(27029)

# Parametri rolling
len_rolling = args.len_rolling
len_rolling_str = f"{args.len_rolling:03d}"
output_mode = args.output
N_stocks = args.N_stocks
training_size = args.training_size
Q = N_stocks / training_size
Q_str = f"{Q:.2f}"

if args.cluster:
    parent_dir = f"Jobs/Q_{Q_str}"
else:
    parent_dir = f"Q_{Q_str}"
os.makedirs(parent_dir, exist_ok=True)

# Crea la sottocartella Rolling_... dentro Q=...
output_dir = os.path.join(parent_dir, f"Rolling_{len_rolling_str}")
os.makedirs(output_dir, exist_ok=True)

# Carica i dati rolling
X_train, X_validation, X_test = FCA.load_stock_data_rolling(file_name="returns_data_1060.csv", 
                                                            train_size= training_size, 
                                                            N_stocks= N_stocks, 
                                                            len_rolling=len_rolling)
Oracle_train, Oracle_validation, Oracle_test = FCA.load_stock_data_rolling(file_name="oracle_returns_data_1060.csv", 
                                                                           train_size= training_size, 
                                                                           N_stocks= N_stocks,
                                                                           len_rolling=len_rolling)

print("Train shape:", X_train.shape)
print("Validation shape:", X_validation.shape)
print("Test shape:", X_test.shape)

# Calcola le performance rolling
save_path_temp = os.path.join(output_dir, "risultati_rolling_temp.pkl")

rolling_performance_dict, rolling_weights_dict = FCA.Compute_Performances_Rolling(
    X_train, X_validation, X_test, Oracle_train, Oracle_validation, Oracle_test, 
    OUTPUT = output_mode, 
    pathfilename_temp = save_path_temp, 
    
)

# Salva i risultati nella directory corretta
save_path = os.path.join(output_dir, "risultati_rolling.pkl")
save_path_weights = os.path.join(output_dir, "risultati_rolling_weights.pkl")
FCA.save_performance_dict(rolling_performance_dict, filename=save_path)
FCA.save_performance_dict(rolling_weights_dict, filename=save_path_weights)