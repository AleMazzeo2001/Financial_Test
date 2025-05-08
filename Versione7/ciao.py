import numpy as np 
import Sigma_Estimation as SE
import financial_test as FCA
import pyRMT as rmt

len_rolling = 250
X_train_3D, X_validation_3D, X_test_3D = FCA.load_stock_data_rolling(file_name="returns_data_1060.csv", len_rolling=len_rolling)
print("Train shape:", X_train_3D.shape)
print("Validation shape:", X_validation_3D.shape)
print("Test shape:", X_test_3D.shape)


Oracle_train_3D, Oracle_validation_3D, Oracle_test_3D = FCA.load_stock_data_rolling(file_name="oracle_returns_data_1060.csv", len_rolling=len_rolling)

for i in range(X_train_3D.shape[0]):
    X_train = X_train_3D[i]
    X_validation = X_validation_3D[i]
    X_test = X_test_3D[i]
    Oracle_train = Oracle_train_3D[i]
    Oracle_validation = Oracle_validation_3D[i]
    Oracle_test = Oracle_test_3D[i]

    E_sample = SE.Sample_Covariance(X_train)

    E_shrunk, best_weight = SE.compute_best_shrinkage_covariance(
        Sigma=E_sample,
        X_train=X_train,
        X_val=X_validation,
        Oracle_train_std=FCA.standardize_returns_oracle(Oracle_train),
        X_train_std=FCA.standardize_returns_oracle(X_train),
        shrinkage_type="diagonal",
        method="Sample_",
        strategy="min_var_",
    )

    E_rie = SE.RIE_Estimator(X_train)
    E_shrunk, best_weight = SE.compute_best_shrinkage_covariance(
        Sigma=E_rie,
        X_train=X_train,
        X_val=X_validation,
        Oracle_train_std=FCA.standardize_returns_oracle(Oracle_train),
        X_train_std=FCA.standardize_returns_oracle(X_train),
        shrinkage_type="diagonal",
        method="Rie____",
        strategy="min_var_",
    )




