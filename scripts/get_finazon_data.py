import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import time

from finazon_grpc_python.time_series_service import TimeSeriesService, GetTimeSeriesRequest
from finazon_grpc_python.common.errors import FinazonGrpcRequestError
# Gloabl api_key, hidden for github
with open('scripts/finazon_key.txt', 'r') as file:
    api_key = file.read()
    
# Helpers
def unix_to_datetime(unix):
    return datetime.fromtimestamp(unix)
def datetime_to_unix(date_str):
    date_format = '%m-%d-%Y %H:%M'
    dt = datetime.strptime(date_str, date_format)
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp
def combine_excel_files(folder_path, output_file):
    """
    After grabbing data through Finazon combine the data into a overall aggregate file
    """
    def finazon_to_unix(date_str):
        date_format = '%Y-%m-%d %H:%M:%S'
        dt = datetime.strptime(date_str, date_format)
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp
    
    excel_files = [file for file in os.listdir(folder_path) if file.endswith(('.csv'))]
    dfs = []

    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df["Unix"] = combined_df["Timestamp"].apply(finazon_to_unix) # Add unix timestamp because it's an easier format to work with
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV file saved as {output_file}")
def plotWeekData(df, line=True):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    fig, ax1 = plt.subplots(figsize=(14, 7))

    if line:
        for day, day_data in df.groupby(df['Timestamp'].dt.date):
            ax1.plot(day_data['Timestamp'], day_data['Open'], label='Open', color='blue', linewidth=1)
            ax1.plot(day_data['Timestamp'], day_data['Close'], label='Close', color='red', linewidth=1)
    else:
        ax1.scatter(df['Timestamp'], df['Open'], label='Open', color='blue', s=10)
        ax1.scatter(df['Timestamp'], df['Close'], label='Close', color='red', s=10)

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Prices Over Time')
    ax1.legend(loc='upper left')

    # Create a second y axis for the volume
    ax2 = ax1.twinx()
    for day, day_data in df.groupby(df['Timestamp'].dt.date):
        ax2.plot(day_data['Timestamp'], day_data['Volume'], label='Volume', color='purple', linewidth=1, linestyle='--')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.show()
def getMinuteStockData(point, tick):
    try:
        service = TimeSeriesService(api_key) # Moved this from global after "Cannot invoke RPC on closed channel"
        end = point + 24 * 60 * 60 #Point plus a day

        request = GetTimeSeriesRequest(
            start_at=point,
            end_at=end,
            ticker=tick,
            dataset='sip_non_pro',
            interval='1m',
            page_size=1000,
            prepost=True
        )
        response = service.get_time_series(request)
        data = []
        for item in response.result:
            data.append([item.timestamp, item.open, item.high, item.low, item.close, item.volume])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        return df
    
    except FinazonGrpcRequestError as e:
        print(f'Received error, code: {e.code}, message: {e.message}')
        return None
def getDataInTimeFrame(start, end, tick, path):
    """
        start:(datetime),
        end:(datetime),
        tick:(str) i.e. AAPL,
        path:(str) path to save folder

        Returns minute stock data of given ticker, saves data of a day in the given folder, 
    """
    point = datetime_to_unix(start)
    end = datetime_to_unix(end)
    error_ctr = 0

    while point <= end:
        print(point)
        df = getMinuteStockData(point, tick)

        if isinstance(df, pd.DataFrame):
            file_name = f"{path}{str(unix_to_datetime(point))[:10]}.csv"
            df.to_csv(file_name, index=False)
            point = point + 86400 # Add a day, not accounting for weekends because prepost
            time.sleep(3)
            error_ctr = 0
        else:
            if error_ctr < 5: # Only get 5 api calls a minute, if it is stalled this long then there must be another error going on
                time.sleep(15.5)
                error_ctr += 1
            else:
                raise Exception("It broke:", df, "Current Point:", point, " | ", unix_to_datetime(point))
                break
def getCurrentPrice(tick):
    """
    Get the current price of a stock
    For testing backend integration
    """
    service = TimeSeriesService(api_key)
    request = GetTimeSeriesRequest(
        ticker=tick,
        start_at=int(time.time() - (24 * 60 * 60)),
        end_at=int(time.time()),
        interval='1m',
        page_size=1000,
        dataset='sip_non_pro',
        prepost=True
    )
    response = service.get_time_series(request)
    return(response.result[-1].close)
        
def main():
    start_date = '08-16-2023 00:00'
    end_date = '09-19-2024 00:00'
    ticker = 'RVTY'
    path = "data/" + ticker + "/raws/" + ticker + "_"

    # getMinuteStockData TESTING
    # df = getMinuteStockData(datetime_to_unix(start_date), ticker)
    # df.to_csv("data/" + ticker + "/raws/Minute/" + ticker + "_" + start_date[0:start_date.find(" ")]+".csv", index=False)

    # getDataInTimeFrame 
    getDataInTimeFrame(start_date, end_date, ticker, path)

# main()
getCurrentPrice('AAPL')
# combine_excel_files("data/AMZN/raws/", "data/AMZN/aggregate/AMZN_agg.csv")