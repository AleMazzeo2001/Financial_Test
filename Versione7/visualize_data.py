import argparse
import os
import sys
FCA_path = os.path.expanduser("~/Desktop/UCL/CODE/Versione7")
sys.path.append(FCA_path)

import financial_test as FCA
import pyRMT as rmt

def main():
    parser = argparse.ArgumentParser(description="Analizza le performance rolling salvate.")
    parser.add_argument("filename", type=str, help="Nome del file .pkl con i risultati delle performance")
    parser.add_argument("--output", type=str, default=None, help="Tipo di output grafico (es. 'Multiple_Boxplot')")
    parser.add_argument("--save", type=bool, default=None, help="Salva i grafici in formato PNG")
    parser.add_argument("--len_rolling", type=int, default=60, help="Lunghezza della finestra rolling")
    parser.add_argument("--N_stocks", type=int, default=400, help="Numero di azioni")
    parser.add_argument("--training_size", type=int, default=800, help="Dimensione del training set")

    args = parser.parse_args()

    N_stocks = args.N_stocks
    training_size = args.training_size
    len_rolling = args.len_rolling
    Q = N_stocks/ training_size
    print("TUA MAMMA", args.len_rolling)

    FCA.load_and_summarize_performance(args.filename, OUTPUT=args.output, Save=args.save, len_rolling=args.len_rolling, Q=Q)

if __name__ == "__main__":
    main()