import pandas as pd
from datetime import datetime
import os
import time

"""
Momentum:
    * If we try use this in a day trading context we should try to find points where we see a .5%-1% (maybe smaller) increase and find the beginning of that
        * Basically local minimum identification
    * Over what period of time?, 
        * Within an hour, 2 hours, etc.
        * Should growth be sustained or spike?
        * Should surrounding points be similarly incentivized?
    * 
"""
def unix_to_datetime(unix):
    return datetime.fromtimestamp(unix)
def find_nearest_date(df, target_date):
    """
    Find the index of the nearest date in the DataFrame to the target date.
    """
    nearest_index = (df["Timestamp"] - target_date).abs().idxmin()
    return nearest_index
def growthDF(df, time_window):
    """
    Params:
    df: DataFrame containing 'Unix' and 'Close' columns
    time_window: Dictionary of time windows (in seconds) for growth calculation

    Returns:
    growth_df: DataFrame containing start times and growth percentages for different time windows
    """
    growth_df = pd.DataFrame(columns=["Unix", "Timestamp"] + list(time_window.keys()))
    
    data = {
        "unix": [],
        "tstamp": [],
    }
    
    for window in time_window:
        data[window] = []
    
    for idx, row in df.iterrows():
        data["unix"].append(row["Unix"])
        data["tstamp"].append(unix_to_datetime(row["Unix"]))
        if idx % 1000 == 0:
            print(unix_to_datetime(row["Unix"]))

        for window in time_window:
            target_date = unix_to_datetime(row["Unix"] + window)
            nearest_index = find_nearest_date(df, target_date)
            nearest_row = df.loc[nearest_index]
            nearest_date = nearest_row["Timestamp"]
            
            if abs((target_date - nearest_date).total_seconds()) < (0.1 * window):
                price_change = (nearest_row["Close"] - row["Close"]) / row["Close"]
                data[window].append(price_change)
            else:
                data[window].append(-1)
    
    growth_df["Unix"] = data["unix"]
    growth_df["Timestamp"] = data["tstamp"]
    
    for window in time_window:
        growth_df[window] = data[window]
    
    return growth_df
def getGrowthDF(ticker):
    """
    Takes in ticker corresponding to pulled data, saves csv containing growth in interval data
    """
    time_window = {
        15*60: "15min",
        30*60: "30min",
        60*60: "1hr",
        120*60: "2hr",
        180*60: "3hr"
    }

    df = pd.read_csv('data/' + ticker +'/aggregate/' + ticker +'_agg.csv')
    df = df[["Unix","Timestamp","Close"]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    growth_df = growthDF(df, time_window)
    growth_df.to_csv("data/" + ticker + "/aggregate/" + ticker + "_growth.csv", index=False)
    print(growth_df)
def instancesOfGrowth(file_path):
    """
    Testing to see what percentage gains are normal in short time periods
    """
    gdf = pd.read_csv(file_path)
    gdf.loc[gdf["900"].idxmax(), "Timestamp"]
    # print("Max 900:", max(gdf["900"]), "Timestamp", gdf.loc[gdf["900"].idxmax(), "Timestamp"])
    # print("Max 1800:", max(gdf["1800"]), "Timestamp", max(range(len(gdf["1800"]))))
    # print("Max 3600:", max(gdf["3600"]), "Timestamp", max(range(len(gdf["3600"]))))
    # print("Max 7200:", max(gdf["7200"]), "Timestamp", max(range(len(gdf["7200"]))))
    # print("Max 10800:", max(gdf["10800"]), "Timestamp", max(range(len(gdf["10800"]))))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
    for i in [0.1, 0.01, 0.005]:
        print("     900   |", i, "instances",  len(gdf[gdf["900"] > i]))
        print("     1800  |", i, "instances",  len(gdf[gdf["1800"] > i]))
        print("     3600  |", i, "instances",  len(gdf[gdf["3600"] > i]))
        print("     7200  |", i, "instances",  len(gdf[gdf["7200"] > i]))
        print("     10800 |", i, "instances",  len(gdf[gdf["10800"] > i]))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
def combineGrowthAggregate(path, ticker, agg_name, growth_name):
    agg_df = pd.read_csv(path + agg_name)
    growth_df = pd.read_csv(path + growth_name)

    agg_df.join(growth_df.set_index('Unix'), on="Unix", lsuffix='_agg', rsuffix='_growth')
    agg_df.to_csv(path + ticker + "_complete.csv", index=False)

# combineGrowthAggregate("data/AMZN/aggregate/", "AMZN", "AMZN_agg.csv", "AMZN_growth.csv")
instancesOfGrowth("scripts/data/AMZN/aggregate/AMZN_growth.csv")