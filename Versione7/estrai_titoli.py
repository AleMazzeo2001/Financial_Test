import pandas as pd

def download_sp500_table(output_path="sp500_companies.csv"):
    """
    Scarica la tabella dei componenti dell'S&P 500 da Wikipedia e la salva come CSV.
    
    Parametri:
        output_path (str): percorso del file CSV di output.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print("Scarico la tabella da Wikipedia...")
    tables = pd.read_html(url)
    
    # La prima tabella è quella dei componenti dell'S&P 500
    sp500_df = tables[0]
    
    # Salva il file CSV
    sp500_df.to_csv(output_path, index=False)
    print(f"Tabella salvata in: {output_path}")

# Scarica la tabella dei componenti dell'S&P 500
#download_sp500_table()

# Step 1: Leggi i ticker dal file returns_data_1060.csv
df = pd.read_csv("returns_data_1060.csv", nrows=0)
tickers = [col for col in df.columns if col.lower() != "date"]
tickers = tickers[:400]

# Step 2: Carica la tabella S&P 500 con Symbol e GICS Sector da file locale
sp500_df = pd.read_csv("sp500_companies.csv")  # <-- questo file lo scarichi tu da Wikipedia

# Step 3: Crea dizionario {ticker: settore}
sector_map = dict(zip(sp500_df['Symbol'].str.upper(), sp500_df['GICS Sector']))

# Step 4: Crea lista con settore associato per ogni ticker (o 'Unknown' se non trovato)
output = []
for ticker in tickers:
    sector = sector_map.get(ticker.upper(), "Unknown")
    output.append((ticker, sector))

# Step 5: Salva su file sia i ticker che i settori
output_df = pd.DataFrame(output, columns=["Ticker", "Sector"])
output_df.to_csv("tickers_with_sectors.csv", index=False)

print("File salvato: tickers_with_sectors.csv")