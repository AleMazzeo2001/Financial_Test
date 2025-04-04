import numpy as np
import matplotlib.pyplot as plt
import financial_test as FCA

# Check Standardization 
""" 
X_train, X_test=FCA.load_stock_data(file_name="log_returns_data_1060.csv")
X_train_std, std_daily, std_stocks=FCA.standardize_returns(X_train)
diff=X_train-X_train_std

#print("Media colonne dopo standardizzazione:", np.mean(X_train_std, axis=0))
print("Media righe dopo standardizzazione:", np.mean(X_train_std, axis=1))
#print("Deviazione standard colonne:", np.std(X_train_std, axis=0))
print("Deviazione standard righe:", np.std(X_train_std, axis=1))
print(f"X_train:{np.linalg.norm(np.mean(X_train, axis=1))}")
print(f"X_std:{np.linalg.norm(np.mean(X_train_std, axis=1))}")

time=np.arange(1, 801, 1)
plt.plot(time, X_train[0,:], label="dati")
plt.plot(time, X_train[1,:], label="dati")
plt.plot(time, X_train[2,:], label="dati")
plt.title("Dati non standardizzati")
plt.legend()
plt.show()

plt.plot(time, X_train_std[0,:])
plt.plot(time, X_train_std[1,:])
plt.plot(time, X_train_std[2,:])
plt.title("Dati standardizzati")
plt.show()

"""

# define parameters
T_train = 800
T_out = 60
n_sets = 50
T_tot = n_sets * T_out


#C_true =  np.diag(np.arange(1, 401))
C_true = np.eye(400)
mean = np.zeros(400)
print(C_true.shape)
    
simulated_data_train =FCA.generate_dataset(C_true, T_train, 1, type="Gaussian")
simulated_data_test= FCA.generate_dataset(C_true, T_out, n_sets, type="Gaussian")
print(simulated_data_train[0,:])
simulated_data_train = np.squeeze(simulated_data_train)








#FCA.Compute_Performances(simulated_data_train, simulated_data_test, OUTPUT=None)










