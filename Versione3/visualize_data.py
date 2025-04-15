import argparse
import financial_test as FCA

def main():
    parser = argparse.ArgumentParser(description="Analizza le performance rolling salvate.")
    parser.add_argument("filename", type=str, help="Nome del file .pkl con i risultati delle performance")
    parser.add_argument("--output", type=str, default=None, help="Tipo di output grafico (es. 'Multiple_Boxplot')")

    args = parser.parse_args()

    FCA.load_and_summarize_performance(args.filename, OUTPUT=args.output)

if __name__ == "__main__":
    main()