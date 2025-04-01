import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime

# Parametri

test_size = 60

# Otteniamo la lista dei titoli dello S&P 500
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]["Symbol"].tolist()
    return tickers

# Scarica dati da Yahoo Finance
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
    return data

# Definiamo il range temporale
#end_date = datetime.today()
#start_date = end_date - timedelta(days=total_days)
start_date  = datetime.datetime(2006, 1, 1) #2007 andava abbastanza bene
end_date  = datetime.datetime(2025, 1, 1)


# Prendiamo i tickers e aggiungiamo titoli extra
sp500_tickers = get_sp500_tickers()
# Prendiamo i tickers e aggiungiamo titoli extra
sp500_tickers = get_sp500_tickers()
extra_tickers = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",  # Big Tech
    "META", "NVDA", "BRK-B", "JPM", "V",  # Finanza & Tech
    "UNH", "XOM", "JNJ", "WMT", "PG",  # Sanità, Energia & Retail
    "MA", "HD", "DIS", "PYPL", "NFLX",  # Pagamenti, Retail & Entertainment
]    # Aggiungiamo alcuni titoli noti


all_tickers = list(set(sp500_tickers + extra_tickers))

print("Numero di Titoli: ", len(all_tickers))




# Crea una lista per raccogliere i dati di tutti i ticker
#log_returns_df = pd.DataFrame() => divisione per NaN, tanto non mi serve
returns_df = pd.DataFrame()
oracle_returns_df = pd.DataFrame()

# Itera sui ticker e raccogli i log-ritorni
for name in all_tickers:
    name_data = yf.Ticker(name).history(start=start_date, end=end_date)

    #log-returns
    #name_log_returns = np.log(name_data['Close'] / name_data['Close'].shift(1)).interpolate(method='time').dropna()
    #log_returns_df[name] = name_log_returns

    #returns
    name_returns = name_data['Close'].pct_change().interpolate(method='time').dropna()
    returns_df[name] = name_returns

    #Oracle prices
    T_out = 60  # Numero di periodi nel lag
    name_returns = name_data['Close'].pct_change(periods=T_out).interpolate(method='time').dropna()
    oracle_returns_df[name] = name_returns

# Remove columns with NaN values from log_returns_df
#log_returns_df.dropna(axis=1, how='any', inplace=True)

# Remove columns with NaN values from returns_df
returns_df.dropna(axis=1, how='any', inplace=True)

# Remove columns with NaN values from oracle_returns_df
oracle_returns_df.dropna(axis=1, how='any', inplace=True)

#log_has_nan = log_returns_df.isna().any().any()
returns_has_nan = returns_df.isna().any().any()
oracle_has_nan = oracle_returns_df.isna().any().any()
#print("log_returns_df has NaN values:", log_has_nan)
print("returns_df has NaN values:", returns_has_nan)
print("oracle_returns_df has NaN values:", oracle_has_nan)


#print(log_returns_df.shape)
print(returns_df.shape)
print(oracle_returns_df.shape)

# Tieni solo i primi 1060 dati (righe)
#log_returns_df = log_returns_df.iloc[:1060]
#returns_df = returns_df.iloc[:1060]
#oracle_returns_df = oracle_returns_df.iloc[:1060]



# Salva il DataFrame ridotto in un file CSV
#log_returns_df.to_csv("log_returns_data_1060.csv", index=True)
returns_df.to_csv("returns_data_1060.csv", index=True)
oracle_returns_df.to_csv("oracle_returns_data_1060.csv", index=True)
