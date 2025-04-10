import numpy as np
import matplotlib.pyplot as plt
import financial_test as FCA
import pyRMT as rmt
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.expanduser("~/Desktop/UCL/CODE/Triangulated_Maximally_Filtered_Graph")
    )
)
from TMFG_core import *



def main():

    # define parameters
    N = 400
    T = 800
    T_out = 60

    # Load the stock data
    X_train, X_test = FCA.load_stock_data(file_name="returns_data_1060.csv")

    FCA.Compute_Performances(X_train, X_test, OUTPUT="Multiple_Boxplot", Compute_MI=False)


def check():

    print("\n\n--CHECK DATA--\n\n")

    # Load the stock data
    X_train, X_test = FCA.load_stock_data()
    print("Dimensioni Train:", X_train.shape)
    print("Dimensioni Test:", X_test.shape)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")

    # Standardize the returns
    X_train, std, std_col = FCA.standardize_returns(X_train)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train dopo la standardizzazione!")

    # Oracle data
    path="/Users/alessandromazzeo/Desktop/UCL/CODE/"

    Oracle_train, Oracle_test = FCA.load_stock_data(
        file_name=path+"oracle_returns_data_1060.csv"
    )
    print("Dimensioni Oracle Train:", Oracle_train.shape)
    print("Dimensioni Oracle Test:", Oracle_test.shape)
    if np.any(np.isnan(X_train)):
        print("Ci sono NaN in X_train!")
    if np.any(np.isnan(X_test)):
        print("Ci sono NaN in X_test!")


def simulated_data():

    print("\n\n--SIMULATED DATA--\n\n")

    # define parameters
    T_train = 800
    T_out = 60
    n_sets = 50
    T_tot = n_sets * T_out

    # Load the stock data
    #X_train, X_test = FCA.load_stock_data()
    #X_train_std, std_daily, std_stocks = FCA.standardize_returns(X_train)

    #C_true = np.cov(X_train_std)
    C_true=  np.diag(np.arange(1, 401))

    # Generate simulated data
    simulated_data_train =FCA.generate_dataset(C_true, T_train, 1, type="Gaussian")
    simulated_data_train = np.squeeze(simulated_data_train)
    simulated_data_test = FCA.generate_dataset(C_true, T_out, n_sets, type="Gaussian")

    print(f"Dimensioni Dati Simulati Train", simulated_data_train.shape)
    # print(f"Dimensioni Dati Simulati Test", simulated_data_test.shape)
    print(f"Dimensioni Covarianza", C_true.shape)

    # Compute_Performances(simulated_data_train, simulated_data_test, OUTPUT="Single_Boxplot")

    # Standardize the returns
    X_train_std, std_daily, std_stocks = FCA.standardize_returns(simulated_data_train)
    #C_true, _,_=FCA.standardize_returns(C_true)
    std_stocks=np.squeeze(std_stocks)

    E_sample = np.cov(X_train_std)
    E_rie = rmt.optimalShrinkage(X_train_std, return_covariance=False, method="rie")
    E_iw = rmt.optimalShrinkage(X_train_std, return_covariance=False, method="iw")
    E_Clipped = rmt.clipped(X_train_std, alpha=0.0, return_covariance=False)

    # TMFG
    model = TMFG()
    corr = np.square(np.corrcoef(X_train_std, rowvar=True))
    E_Sample_TMFG = np.cov(X_train_std)
    _, _, J_TMFG = model.fit_transform(weights=corr, cov=E_Sample_TMFG, output="logo")
    E_TMFG = np.linalg.inv(J_TMFG)

    # Calcolo degli autovalori e ordinamento in ordine decrescente
    print(np.linalg.eigvals(C_true)[:5])
    lambda_C = np.sort(np.linalg.eigvals(C_true))
    
    print(lambda_C[:5])
    lambda_E_sample = np.sort(np.linalg.eigvals(E_sample))* std_stocks
    print(lambda_E_sample.shape) 
    print(std_stocks.shape)
    print(lambda_E_sample[:5])
    print(lambda_E_sample.shape)
    lambda_E_rie = np.sort(np.linalg.eigvals(E_rie))* np.sort(std_stocks)
    lambda_E_iw = np.sort(np.linalg.eigvals(E_iw))* np.sort(std_stocks)
    lambda_E_Clipped = np.sort(np.linalg.eigvals(E_Clipped))* np.sort(std_stocks)
    lambda_E_TMFG = np.sort(np.linalg.eigvals(E_TMFG))* np.sort(std_stocks)

    plt.plot(lambda_C, lambda_C, marker="o", linestyle="-", label="C_True")
    plt.plot(lambda_C, lambda_E_sample, marker="o", linestyle="-", label="E_Sample")
    #plt.plot(lambda_C, lambda_E_rie, marker="o", linestyle="-", label="E_Rie")
    #plt.plot(lambda_C, lambda_E_iw, marker="o", linestyle="-", label="E_IW")
    #plt.plot(lambda_C, lambda_E_Clipped, marker="o", linestyle="-", label="E_Clipped")
    #plt.plot(lambda_C, lambda_E_TMFG, marker="o", linestyle="-", label="E_TMFG")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()    
    plt.show()

    FCA.Compute_Performances(simulated_data_train, simulated_data_test, OUTPUT="Multi_Boxplot", Compute_MI=False)




if __name__ == "__main__":
    #check()
    #simulated_data()
    main()



