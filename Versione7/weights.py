import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Carica il dizionario
with open("Q_0.50/Rolling_300/risultati_rolling_weights.pkl", "rb") as f:
    rolling_weights_dict = pickle.load(f)

# Raggruppa le entry per (strategy, method_name)
grouped_data = defaultdict(lambda: {})

for (strategy, method_name, i), w in rolling_weights_dict.items():
    grouped_data[(strategy, method_name)][i] = np.array(w).flatten()  # Converti in array 1D

# Per ogni combinazione (strategy, method_name), produci il grafico
for (strategy, method_name), time_dict in grouped_data.items():
    # Ordina gli step temporali
    sorted_indices = sorted(time_dict.keys())
    data = [time_dict[i] for i in sorted_indices]

    # Calcola le medie
    means = [np.mean(w) for w in data]
    medians = [np.median(w) for w in data]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, positions=sorted_indices, showfliers=False)
    plt.plot(sorted_indices, means, color='red', linestyle='-', marker='o', label='Media dei pesi')
    plt.plot(sorted_indices, medians, color='blue', linestyle='--', marker='x', label='Mediana dei pesi')

    plt.title(f"Distribuzione pesi - Strategy: {strategy}, Method: {method_name}")
    plt.xlabel("Indice temporale (i)")
    plt.ylabel("Peso")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()