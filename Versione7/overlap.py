import numpy as np
import matplotlib.pyplot as plt
import financial_test as FCA
import Sigma_Estimation as SE

N=400
T=800
N_matrices = 1000
C_true = np.eye(N)
def generate_spicy_precision(N, rho=0.5):
    """
    Genera una precision matrix NxN con struttura AR(1)
    """
    # Generiamo la matrice di correlazione AR(1)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = rho ** abs(i - j)
    
    # La precision matrix è semplicemente l'inversa
    J_true = np.linalg.inv(C)
    
    return J_true

# Esempio:
J_true = generate_spicy_precision(N, rho=0.6)
print(J_true)
C_true = np.linalg.inv(J_true)


dataset = FCA.generate_dataset(C_true, T, N_matrices, type = "Gaussian")
print("Dataset generated with shape:", dataset.shape)


def compute_overlap(dataset, J_true, SE):
    """
    dataset: numpy array di shape (numero_matrici, numero_variabili, numero_osservazioni)
    J_true: matrice di precisione vera (numero_variabili, numero_variabili)
    SE: modulo che contiene gli stimatori di covarianza e precisione
    """
    
    numero_matrici, numero_variabili, numero_osservazioni = dataset.shape
    
    # Diagonalizzazione di J_true una volta sola
    eigvals_true, eigvecs_true = np.linalg.eigh(J_true)
    
    # Definiamo gli stimatori: per ciascuno specifichiamo se restituiscono la precision direttamente o la covarianza
    estimators = {
        'Sample':          {'func': SE.Sample_Covariance, 'is_precision': False},
        'RIE':             {'func': SE.RIE_Estimator,     'is_precision': False},
        'RIE_IW':          {'func': SE.RIE_IW_Estimator,  'is_precision': False},
        'Clipped':         {'func': SE.Clipped_Estimator, 'is_precision': False},
        'TMFG':            {'func': lambda X: SE.Fast_TMFG(X)[1], 'is_precision': True}
    }
    
    # Dizionario per contenere gli overlap per ciascun metodo
    overlaps_all = {name: np.zeros((numero_matrici, numero_variabili)) for name in estimators.keys()}
    
    for i in range(numero_matrici):
        X_train = dataset[i]  # shape (numero_variabili, numero_osservazioni)
        
        for name, estimator_info in estimators.items():
            estimator_func = estimator_info['func']
            is_precision = estimator_info['is_precision']
            
            try:
                est = estimator_func(X_train)
            except Exception as e:
                print(f"Errore nello stimatore {name} al passo {i}: {e}")
                continue
            
            # Se restituisce la precision direttamente
            if is_precision:
                precision = est
            else:
                # Altrimenti invertiamo la covarianza stimata
                try:
                    precision = np.linalg.inv(est)
                except np.linalg.LinAlgError:
                    print(f"Warning: matrice singolare per {name} al passo {i}, uso pseudoinversa")
                    precision = np.linalg.pinv(est)
            
            # Diagonalizza la precision
            eigvals_empirical, eigvecs_empirical = np.linalg.eigh(precision)
            
            # Calcolo overlap (normalizziamo per sicurezza)
            for j in range(numero_variabili):
                v_true = eigvecs_true[:, j]
                v_empirical = eigvecs_empirical[:, j]
                v_true /= np.linalg.norm(v_true)
                v_empirical /= np.linalg.norm(v_empirical)
                overlap = N*(np.dot(v_true, v_empirical)**2)
                overlaps_all[name][i, j] = overlap

    # Ora calcoliamo la media su tutte le matrici per ciascun stimatore
    mean_overlaps_all = {name: np.mean(overlaps, axis=0) for name, overlaps in overlaps_all.items()}
    
    # Plot finale
    plt.figure(figsize=(10, 6))
    
    for name, mean_overlaps in mean_overlaps_all.items():
        plt.plot(mean_overlaps, marker='o', linestyle='-', label=name)
    
    plt.axhline(y=N*1.0, color='r', linestyle='--', label='Perfect overlap')
    plt.xlabel('Eigenvector index')
    plt.ylabel('Mean squared overlap')
    plt.title('Mean squared overlap vs true eigenvectors')
    plt.legend()
    plt.grid()
    plt.show()
    
    return mean_overlaps_all

_ = compute_overlap(dataset, C_true, SE)