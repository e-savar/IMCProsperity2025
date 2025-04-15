import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the CSV file and parse the time column
df = pd.read_csv("PriceData.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert 'timestamp' to datetime
df = df.sort_values('timestamp')

# Step 2: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['mid_price'], label="Mid Price")
plt.xlabel('Time')
plt.ylabel('Mid Price')
plt.title('Mid Price Over Time')
plt.show()

# Statistical summary
print(df['mid_price'].describe())

# Check for missing data
print(df.isnull().sum())

# Step 3: Feature Engineering
df['MA_5'] = df['mid_price'].rolling(window=5).mean()
df['MA_15'] = df['mid_price'].rolling(window=15).mean()
df['volatility'] = df['mid_price'].rolling(window=5).std()
df['return'] = df['mid_price'].pct_change()

# Z-Score
mean_rolling = df['mid_price'].rolling(window=10).mean()
std_rolling = df['mid_price'].rolling(window=10).std()
df['zscore'] = (df['mid_price'] - mean_rolling) / std_rolling

# Step 4: Strategy Signals
df['signal'] = 0

# Moving Average Crossover
df.loc[df['MA_5'] > df['MA_15'], 'signal'] = 1
df.loc[df['MA_5'] < df['MA_15'], 'signal'] = -1

# Z-score overrides (strong mean reversion when |z| > 1)
df.loc[df['zscore'] > 2, 'signal'] = -1  # Overbought → sell
df.loc[df['zscore'] < -2, 'signal'] = 1  # Oversold → buy

df['signal'] = df['signal'].shift(1)  # Apply signal on next time step

# Step 5: Strategy Returns
df['strategy_return'] = df['signal'] * df['return']
df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()

# Step 6: Performance Plot
plt.figure(figsize=(10, 6))
df['cumulative_return'].plot()
plt.title('Cumulative Return of Combined Strategy')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

# Step 7: Sharpe Ratio
sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std()
print(f'Sharpe Ratio: {sharpe_ratio:.4f}')

# Step 8: Backtesting - Simulating Portfolio Value
initial_cash = 100000
cash = initial_cash
position = 0
portfolio_value = []

for i in range(len(df)):
    signal = df['signal'].iloc[i]
    price = df['mid_price'].iloc[i]
    
    if signal == 1 and cash > price:
        position = cash // price
        cash -= position * price
    elif signal == -1 and position > 0:
        cash += position * price
        position = 0
    
    portfolio_value.append(cash + position * price)

# Final Portfolio Value Plot
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], portfolio_value, label="Simulated Portfolio Value")
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.grid(True)
plt.legend()
plt.show()

print(f"Final portfolio value: ${portfolio_value[-1]:,.2f}")
