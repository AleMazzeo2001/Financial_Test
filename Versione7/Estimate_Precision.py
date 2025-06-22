import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import financial_test as FCA
import Sigma_Estimation as SE
import Precision_Block
import os

def Relative_Frobenius(C_true, Sigma_estimate):
    return np.linalg.norm(C_true - Sigma_estimate, 'fro') / np.linalg.norm(C_true, 'fro')

def compute_relative_frobenius(dataset, J_true, SE):
    estimators = {
        'Sample':    {'func': SE.Sample_Covariance, 'is_covariance': True},
        'RIE':       {'func': SE.RIE_Estimator,     'is_covariance': True},
        'RIE_IW':    {'func': SE.RIE_IW_Estimator,  'is_covariance': True},
        'Clipped':   {'func': SE.Clipped_Estimator, 'is_covariance': True},
        'Kendall': {'func': SE.Kendall_Estimator, 'is_covariance': True},
        'TMFG':      {'func': lambda X: SE.Fast_TMFG(X)[1], 'is_covariance': False}
    }

    num_matrices = dataset.shape[0]
    relative_frobenius_all = {name: np.zeros(num_matrices) for name in estimators}

    for i in range(num_matrices):
        X_train = dataset[i]
        for name, est in estimators.items():
            J_est = est['func'](X_train)
            if est['is_covariance']:
                J_est = np.linalg.pinv(J_est)
            relative_frobenius_all[name][i] = Relative_Frobenius(J_true, J_est)

    r1_mean = {name: np.mean(v) for name, v in relative_frobenius_all.items()}
    r1_std = {name: np.std(v) for name, v in relative_frobenius_all.items()}

    return r1_mean, r1_std

def frobenius_performance_vs_q(N=400, T_list=None, N_matrices=30, sparsity=0.1, output_file="frobenius_results.csv"):
    if T_list is None:
        T_list = list(range(100, 810, 10))
        
    # Preparazione del file CSV
    results = []

    for T in T_list:
        q = N / T
        print(f"\n=== Calcolo per T = {T}, q = {q:.3f} ===")
        J_true, C_true = Precision_Block.generate_sparse_precision_matrix(N, sparsity)
        dataset = FCA.generate_dataset(C_true, T, N_matrices, type="Gaussian")
        r1_mean, r1_std = compute_relative_frobenius(dataset, J_true, SE)

        print(f"{'Estimator':<12} | {'Mean':>10} | {'Std Dev':>10} | {'CV (%)':>10}")
        print("-" * 50)
        for method in r1_mean:
            results.append({
                "method": method,
                "mean": r1_mean[method],
                "std": r1_std[method],
                "q": q
            })

            cv = 100 * r1_std[method]/ r1_mean[method] if r1_mean[method] != 0 else np.nan
            print(f"{method:<12} | {r1_mean[method]:10.4f} | {r1_std[method]:10.4f} | {r1_std[method]:10.2f}")
            #print(f"Metodo: {method}, Media: {r1_mean[method]:.4f}, Std: {r1_std[method]:.4f}")
        print("-" * 50)
        

    # Salva su CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        print(f"\nRisultati  parziali salvati in {output_file}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\nRisultati  finali salvati in {output_file}")

    return df_results

def plot_frobenius_results(results_csv="frobenius_results.csv", output_pdf="J_Reconstruction.pdf"):
    df = pd.read_csv(results_csv)

    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    # Plot con errore
    sns.lineplot(data=df, x="q", y="mean", hue="method", marker="o", err_style="bars", err_kws={'capsize': 3}, errorbar=None)
    for method in df["method"].unique():
        if method =="RIE":
            continue
        sub = df[df["method"] == method]
        plt.fill_between(sub["q"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                         alpha=0.2, label=f"{method} ± std")

    plt.xlabel("q = N / T")
    plt.ylabel("Relative Frobenius Distance")
    plt.yscale("log")
    plt.title("Precision Matrix Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf)
    print(f"Plot salvato in {output_pdf}")
    plt.close()

if __name__ == "__main__":
    N = 400
    T_values = list(range(100, 801, 10))
    #T_values = [220, 230, 240]  # Example values
    N_matrices = 30
    sparsity = 0.1

    results_csv = "frobenius_results.csv"
    plot_pdf = "J_Reconstruction.pdf"

    #df_all = frobenius_performance_vs_q(N=N, T_list=T_values, N_matrices=N_matrices, sparsity=sparsity, output_file=results_csv)
    plot_frobenius_results(results_csv=results_csv, output_pdf=plot_pdf)





