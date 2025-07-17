#!/usr/bin/env python
# download_expanded_data.py
# =========================================================
# Robustly downloads an expanded list of economic indicators
# and saves the clean data to a CSV file.
# =========================================================

import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import sys

# --- Configuration ---
# All the indicators you requested
EXPANDED_ECONOMIC_INDICATORS = {
    # Market Indices
    "NASDAQ": "^IXIC",
    "Russell2000": "^RUT",
    "DowJones": "^DJI",
    
    # Volatility
    "VIX": "^VIX",
    "VIX9D": "^VIX9D",
    "VXN": "^VXN",
    
    # Interest Rates & Bonds
    # Note: Using different tickers for clarity as ^TYX was duplicated.
    # 2Y Treasury is ^IRX, 10Y is ^TNX, 30Y is ^TYX.
    # Using LQD as a proxy for Investment Grade Corporate Bond Yield behavior.
    "2Y_Treasury": "^IRX",
    "10Y_Treasury": "^TNX",
    "30Y_Treasury": "^TYX", 
    "Corporate_Bond_ETF": "LQD",
    
    # Sector ETFs
    "Banking": "KBE",
    "Regional_Banks": "KRE", 
    "REITs": "VNQ",
    "Utilities": "XLU",
    "Technology": "XLK",
    
    # Economic Health / Commodities
    "Dollar_Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "Oil": "CL=F",
    
    # Dividend-focused ETFs
    "Dividend_ETF": "DVY",
    "High_Dividend": "HDV",
    "Dividend_Aristocrats": "NOBL"
}

START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 5
RETRY_DELAY = 5
CSV_OUTPUT_FILE = "expanded_economic_data.csv"


def robust_download(tickers_list, start, end, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """
    Attempts to download data from yfinance with a retry mechanism.
    """
    print(f"--- Attempting to download data for {len(tickers_list)} tickers from {start} to {end} ---")
    for i in range(retries):
        try:
            data = yf.download(tickers_list, start=start, end=end, progress=False)
            if data.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            
            print("✅ Initial download successful!")
            return data
            
        except Exception as e:
            print(f"⚠️ Download attempt {i+1} of {retries} failed: {e}")
            if i < retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("❌ All download attempts failed. Exiting.")
                sys.exit(1)

def main():
    """
    Main function to execute the data processing pipeline.
    """
    # Create a list of unique ticker symbols to download
    unique_tickers = list(set(EXPANDED_ECONOMIC_INDICATORS.values()))
    
    # 1. Download data using the robust function
    raw_data = robust_download(unique_tickers, START_DATE, END_DATE)
    
    # 2. Clean and process the data
    print("\n--- Processing and cleaning data ---")
    close_prices = raw_data['Close']
    
    # 3. Identify and handle any failed downloads
    failed_tickers = close_prices.columns[close_prices.isna().all()].tolist()
    if failed_tickers:
        print(f"⚠️ Failed to retrieve data for the following tickers: {failed_tickers}")
        close_prices = close_prices.drop(columns=failed_tickers)
        print("These tickers have been removed from the final dataset.")
        
    print(f"Successfully retrieved data for {close_prices.shape[1]} tickers.")

    # 4. Rename columns to be more friendly for the model
    # Create a reverse mapping from ticker symbol to friendly name
    ticker_to_name = {v: k for k, v in EXPANDED_ECONOMIC_INDICATORS.items()}
    close_prices = close_prices.rename(columns=ticker_to_name)
    
    # 5. Fill any remaining missing values
    if close_prices.isnull().sum().sum() > 0:
        print("Missing values found for some dates. Filling them using forward-fill.")
        close_prices.ffill(inplace=True)
        close_prices.bfill(inplace=True)
    
    print("Data cleaning complete.")
    
    # 6. Save the clean, model-ready data to a CSV file
    print(f"\n--- Saving clean model-ready data to '{CSV_OUTPUT_FILE}' ---")
    close_prices.to_csv(CSV_OUTPUT_FILE)
    
    print("✅ All tasks complete. Your expanded data is ready!")
    print(f"\nFinal data preview ('{CSV_OUTPUT_FILE}'):\n{close_prices.tail()}")


if __name__ == "__main__":
    main()
