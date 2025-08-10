import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_stock_data(df, ticker, days, data_points):
    """
    Create a styled stock price plot using matplotlib.
    
    Args:
        df: DataFrame with Date and Close columns
        ticker: Stock symbol 
        days: Number of days selected by user
        data_points: Actual number of trading days in the data
    
    Returns:
        matplotlib figure object
    """
    # Clear any existing plots
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert Date to datetime if it isn't already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Plot line and points
    ax.plot(df['Date'], df['Close'], 
            color='#2E86C1', 
            linewidth=2, 
            alpha=0.8,
            label='Close Price')
    ax.scatter(df['Date'], df['Close'], 
              color='#2E86C1', 
              s=50, 
              alpha=0.6)

    # Calculate moving average if enough data points
    if len(df) > 5:
        ma = df['Close'].rolling(window=5).mean()
        ax.plot(df['Date'], ma, 
                color='#E67E22', 
                linestyle='--', 
                linewidth=1.5,
                alpha=0.8,
                label='5-day MA')

    # Styling
    ax.set_title(f"{ticker.upper()} Stock Price - Last {data_points} Trading Days", 
                 fontsize=14, 
                 pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend
    ax.legend(loc='upper left', frameon=True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # plt.show()
    
    return fig