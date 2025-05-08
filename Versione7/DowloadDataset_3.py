import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime

# ========== PARAMETRI ==========

test_size = 60

# ========== FUNZIONI UTILI ==========

def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]["Symbol"].tolist()
    return tickers

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
    return data

def remove_redundant_series(df, corr_thresh=0.99999, norm_thresh=1e-7):
    redundant = set()
    redundant_pairs = []

    cols = df.columns
    X = df.values.T

    N = len(cols)
    for i in range(N):
        if cols[i] in redundant:
            continue
        for j in range(i + 1, N):
            if cols[j] in redundant:
                continue

            rho = np.corrcoef(X[i], X[j])[0, 1]
            dist = np.sqrt(2 * (1 - rho))
            norm_diff = np.linalg.norm(X[i] - X[j])

            if rho > corr_thresh or dist < 1e-5 or norm_diff < norm_thresh:
                redundant.add(cols[j])
                redundant_pairs.append((cols[i], cols[j]))

    df_filtered = df.drop(columns=redundant)
    return df_filtered, redundant_pairs

# ========== RANGE TEMPORALE ==========

start_date  = datetime.datetime(2006, 6, 1)
end_date  = datetime.datetime(2025, 3, 1)

# ========== TICKERS ==========

sp500_tickers = get_sp500_tickers()
extra_tickers = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
    "META", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "PG",
    "MA", "HD", "DIS", "PYPL", "NFLX",
]

all_tickers = list(set(sp500_tickers + extra_tickers))
print("Numero di Titoli: ", len(all_tickers))

# ========== SCARICAMENTO DATI ==========

returns_df = pd.DataFrame()
oracle_returns_df = pd.DataFrame()

for name in all_tickers:
    name_data = yf.Ticker(name).history(start=start_date, end=end_date)

    name_returns = name_data['Close'].pct_change().interpolate(method='time').dropna()
    returns_df[name] = name_returns

    T_out = 60
    name_returns_oracle = name_data['Close'].pct_change(periods=T_out).interpolate(method='time').dropna()
    oracle_returns_df[name] = name_returns_oracle

returns_df.dropna(axis=1, how='any', inplace=True)
oracle_returns_df.dropna(axis=1, how='any', inplace=True)

log_returns_df = np.log(1 + returns_df)

# ========== RIMOZIONE SERIE RIDONDANTI ==========

log_returns_df, redundant_pairs = remove_redundant_series(log_returns_df)

# Applichiamo la stessa selezione a returns_df e oracle_returns_df
tickers_to_keep = log_returns_df.columns
returns_df = returns_df[tickers_to_keep]
oracle_returns_df = oracle_returns_df[tickers_to_keep]

# ========== OUTPUT ==========

print(f"\nSerie rimosse: {len(redundant_pairs)}")
for a, b in redundant_pairs:
    print(f"{b} rimosso (troppo simile a {a})")

print("Dimensioni finali:")
print("log_returns_df:", log_returns_df.shape)
print("returns_df:", returns_df.shape)
print("oracle_returns_df:", oracle_returns_df.shape)

log_returns_df.to_csv("log_returns_data_1060.csv", index=True)
returns_df.to_csv("returns_data_1060.csv", index=True)
oracle_returns_df.to_csv("oracle_returns_data_1060.csv", index=True)