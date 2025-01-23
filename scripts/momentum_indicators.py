import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_roc(data, window=14):
    roc = ((data - data.shift(window)) / data.shift(window)) * 100
    return roc
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'MACD_Histogram': macd_histogram
    })
def calculate_mfi(data, window=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    
    # Identify positive and negative money flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    # Calculate the money flow ratio and MFI
    positive_mf_sum = positive_flow.rolling(window=window).sum()
    negative_mf_sum = negative_flow.rolling(window=window).sum()
    money_flow_ratio = positive_mf_sum / negative_mf_sum
    
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi
def main(filename, output_path, ohlc="Close"):
    data = pd.read_csv(filename)
    # Add technical indicators
    data['RSI'] = calculate_rsi(data[ohlc])
    data['ROC'] = calculate_roc(data[ohlc])
    data['MFI'] = calculate_mfi(data)
    macd_df = calculate_macd(data[ohlc])
    data = pd.concat([data, macd_df], axis=1)

    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    main("data/AAPL/AAPL_data.csv", "data/AAPL/AAPL_data.csv")