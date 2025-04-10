import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Define a function to process each file
def process_file(file_path):
    # Load the data with market information
    data = pd.read_csv(file_path, sep=";", parse_dates=['timestamp'], index_col='timestamp')
    
    # Step 2: Calculate the mid_price from bid and ask prices if it's not already provided
    data['mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2
    
    # Step 3: Preprocess the data by smoothing using moving average (optional)
    data['smoothed_price'] = data['mid_price'].rolling(window=10).mean()  # 10 is the moving window size
    
    # Step 4: Fit an ARIMA model on the smoothed price data
    model = ARIMA(data['smoothed_price'].dropna(), order=(5, 0, 5))  # ARIMA(p, d, q) order
    model_fit = model.fit()

    # Print model summary
    print(f"Model summary for {file_path}:")
    print(model_fit.summary())
    
    # Step 5: Forecast the next 10 steps
    forecast_steps = 10
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Ensure forecast is in a format that can be indexed
    if isinstance(forecast, np.ndarray):
        forecast = pd.Series(forecast)

    print(f"Forecasted Prices for {file_path}: {forecast}")
    
    # Step 6: Define a simple trading strategy based on mean reversion
    mean_price = data['smoothed_price'].mean()
    
    # Mean Reversion Strategy: Buy if price is below mean, Sell if price is above mean
    threshold = 0.01  # 1% threshold for mean reversion strategy
    buy_signal = forecast.iloc[0] < (mean_price * (1 - threshold))  # Access the first element safely
    sell_signal = forecast.iloc[0] > (mean_price * (1 + threshold))  # Access the first element safely
    
    # Step 7: Print trading signal
    if buy_signal:
        print(f"Buy Signal for {file_path}")
    elif sell_signal:
        print(f"Sell Signal for {file_path}")
    else:
        print(f"No Signal for {file_path}")
    
    # Step 8: Backtest the strategy
    # Calculate returns based on price data and strategy signals
    data['returns'] = data['mid_price'].pct_change()
    data['strategy'] = np.where(data['mid_price'] > mean_price, 1, -1)  # Long if above mean, short if below
    
    # Calculate strategy returns
    data['strategy_returns'] = data['strategy'] * data['returns']
    data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
    
    # Plot strategy performance
    plt.figure(figsize=(10, 6))
    plt.plot(data['cumulative_strategy_returns'], label=f'Strategy Performance for {file_path}')
    plt.legend()
    plt.title(f"Strategy Performance for {file_path}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.show()

# Step 9: Loop through all CSV files in the folder
folder_path = "price_files"  # Replace this with the actual folder path

# Loop through CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name}...")
        process_file(file_path)
