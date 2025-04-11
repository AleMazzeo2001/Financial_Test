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

np.random.seed(27029) 

X_train , X_test= FCA.load_stock_data_rolling(file_name="returns_data_1060.csv", len_rolling=5)
print(X_train.shape)
print(X_test.shape)

Oracle_train, Oracle_test = FCA.load_stock_data_rolling(file_name="oracle_returns_data_1060.csv", len_rolling=5)
print(X_train.shape)
print(X_test.shape)

rolling_performance_dict = FCA.Compute_Performances_Rolling(X_train, X_test, Oracle_train, Oracle_test, OUTPUT="Multiple_Boxplot")
FCA.save_performance_dict(rolling_performance_dict, filename="risultati_rolling.pkl")

