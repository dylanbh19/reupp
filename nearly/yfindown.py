#!/usr/bin/env python
# download_economic_data.py
# =========================================================
# Robustly downloads, normalizes, plots, and saves
# key economic indicators for model input.
# =========================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import sys

# --- Configuration ---
TICKERS = {
    "VIX": "^VIX",                  # Volatility Index
    "SP500": "^GSPC",               # S&P 500 Market Index
    "InterestRate_10Y": "^TNX",     # 10-Year Treasury Note Yield
    "FinancialSector": "XLF"        # Financial Select Sector SPDR Fund
}
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 5 # Number of times to attempt download
RETRY_DELAY = 5 # Seconds to wait between retries
CSV_OUTPUT_FILE = "economic_data_for_model.csv"
PLOT_OUTPUT_FILE = "economic_indicators_plot.png"

def robust_download(tickers_list, start, end, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """
    Attempts to download data from yfinance with a retry mechanism.
    """
    print(f"--- Attempting to download data for {len(tickers_list)} tickers from {start} to {end} ---")
    for i in range(retries):
        try:
            # Attempt to download the data
            data = yf.download(tickers_list, start=start, end=end, progress=False)
            
            # Check if the download was successful (sometimes it returns an empty df)
            if data.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            
            print("✅ Download successful!")
            return data
            
        except Exception as e:
            print(f"⚠️ Download attempt {i+1} of {retries} failed: {e}")
            if i < retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("❌ All download attempts failed. Exiting.")
                sys.exit(1) # Exit the script if all retries fail

def main():
    """
    Main function to execute the data processing pipeline.
    """
    # 1. Download data using the robust function
    raw_data = robust_download(list(TICKERS.values()), START_DATE, END_DATE)
    
    # 2. Clean and process the data
    print("\n--- Processing and cleaning data ---")
    # We only care about the 'Close' price for our model
    close_prices = raw_data['Close']
    
    # Rename columns to be more friendly
    close_prices = close_prices.rename(columns={v: k for k, v in TICKERS.items()})
    
    # Check for and fill any missing values (e.g., from holidays)
    if close_prices.isnull().sum().sum() > 0:
        print("Missing values found. Filling them using forward-fill.")
        close_prices.ffill(inplace=True)
        # Use back-fill for any remaining NaNs at the start
        close_prices.bfill(inplace=True)
    
    print("Data cleaning complete.")

    # 3. Normalize the data for plotting
    print("\n--- Normalizing data for trend comparison plot ---")
    scaler = MinMaxScaler()
    normalized_prices_array = scaler.fit_transform(close_prices)
    normalized_prices = pd.DataFrame(normalized_prices_array, index=close_prices.index, columns=close_prices.columns)

    # 4. Plot the normalized data
    print(f"--- Generating and saving plot to '{PLOT_OUTPUT_FILE}' ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    normalized_prices.plot(ax=ax)
    
    ax.set_title(f'Normalized Economic Indicators ({START_DATE} to {END_DATE})', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Value (0 to 1)', fontsize=12)
    ax.legend(title='Indicator', fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_FILE)
    # plt.show() # Uncomment this line if you want the plot to display automatically
    print("Plot saved successfully.")

    # 5. Save the clean, non-normalized data for your model
    print(f"\n--- Saving clean model-ready data to '{CSV_OUTPUT_FILE}' ---")
    close_prices.to_csv(CSV_OUTPUT_FILE)
    print("✅ All tasks complete. Your data is ready!")
    print(f"\nFinal data preview:\n{close_prices.tail()}")


if __name__ == "__main__":
    main()
