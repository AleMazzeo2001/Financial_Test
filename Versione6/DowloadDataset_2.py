import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations

# === PARAMETRI ===
start_date = datetime(2006, 6, 1)
end_date   = datetime(2025, 3, 1)
T_out = 60  # Oracle shift
CORR_THRESHOLD = 0.9999999

# === FUNZIONI UTILI ===
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]["Symbol"].tolist()
    # Yahoo usa "-" al posto di "." (es. BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

def download_and_process(tickers, start_date, end_date):
    returns_df = pd.DataFrame()
    oracle_returns_df = pd.DataFrame()
    
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)

            if data.empty or "Close" not in data:
                print(f"⚠️ Nessun dato per {ticker}")
                continue

            # returns
            returns = data['Close'].pct_change().interpolate(method='time').dropna()
            oracle = data['Close'].pct_change(periods=T_out).interpolate(method='time').dropna()

            if returns.isnull().any() or oracle.isnull().any():
                continue

            returns_df[ticker] = returns
            oracle_returns_df[ticker] = oracle

        except Exception as e:
            print(f"Errore con {ticker}: {e}")

    return returns_df, oracle_returns_df

def find_highly_correlated_pairs(df, threshold=CORR_THRESHOLD):
    corr_matrix = df.corr()
    pairs = []
    for i, j in combinations(df.columns, 2):
        corr = corr_matrix.loc[i, j]
        if np.abs(corr) > threshold:
            pairs.append((i, j, corr))
    return pairs

# === TICKERS ===
sp500 = get_sp500_tickers()
extra = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",  # Big Tech
    "META", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "PG",
    "MA", "HD", "DIS", "PYPL", "NFLX",
]
# Uniformiamo il formato
extra = [t.replace(".", "-") for t in extra]

# Rimuovi duplicati e ordina
all_tickers = sorted(set(sp500 + extra))
print("Numero totale di ticker:", len(all_tickers))

# === DOWNLOAD DATI ===
returns_df, oracle_returns_df = download_and_process(all_tickers, start_date, end_date)

# Rimuove colonne con NaN
returns_df.dropna(axis=1, how='any', inplace=True)
oracle_returns_df.dropna(axis=1, how='any', inplace=True)

# Log-returns
log_returns_df = np.log(1 + returns_df)

# === RILEVAZIONE CORRELAZIONI TROPPO ALTE ===
print("\n🔍 Controllo coppie altamente correlate...")
high_corr_pairs = find_highly_correlated_pairs(log_returns_df)
if high_corr_pairs:
    for a, b, corr in high_corr_pairs:
        print(f"{a} & {b} → correlazione: {corr:.10f}")
else:
    print("✅ Nessuna coppia altamente correlata trovata.")

# === ESPORTAZIONE ===
log_returns_df.to_csv("log_returns_data_1060.csv")
returns_df.to_csv("returns_data_1060.csv")
oracle_returns_df.to_csv("oracle_returns_data_1060.csv")

print("\n✅ Tutto fatto. Shape finali:")
print("Log-returns:", log_returns_df.shape)
print("Returns:    ", returns_df.shape)
print("Oracle:     ", oracle_returns_df.shape)