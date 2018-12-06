"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data
import math

def KDJ(symbols, start_date, end_date, n, a, b):
    data = get_data([symbols], pd.date_range(start_date, end_date), colname = 'High')
    data = data[symbols]
    low = get_data([symbols], pd.date_range(start_date, end_date), colname = 'Low')
    data['low'] = low[symbols]
    adjust_close = get_data([symbols], pd.date_range(start_date, end_date))
    data['adjust_close'] = adjust_close[symbols]
    print data
    return data

def calSMA(symbols, start_date, end_date, period):
    data = get_data([symbols], pd.date_range(start_date, end_date))
    data = data[[symbols]]
    data = data.iloc[:]/data.iloc[0]
    for i in range(data.shape[1]):
	data['{} SMA'.format(data.columns.values[i])] = data[data.columns.values[i]].rolling(window = period, min_periods = period).mean()
	data['{} price/SMA ratio'.format(data.columns.values[i])] = data[data.columns.values[i]]/data['{} SMA'.format(data.columns.values[i])]
    return data

def calBB(symbols, start_date, end_date, period):
    data = get_data([symbols], pd.date_range(start_date, end_date))
    data = data[[symbols]]
    data = data.iloc[:]/data.iloc[0]
    for i in range(data.shape[1]):
	data['{} UB'.format(data.columns.values[i])] = data[data.columns.values[i]].rolling(window = period, min_periods = period).mean() + (2 * data[data.columns.values[i]].rolling(window = period, min_periods = period).std())
	data['{} LB'.format(data.columns.values[i])] = data[data.columns.values[i]].rolling(window = period, min_periods = period).mean() - (2 * data[data.columns.values[i]].rolling(window = period, min_periods = period).std())
	data['{} BBP'.format(data.columns.values[i])] = (data[data.columns.values[i]] - data['{} LB'.format(data.columns.values[i])]) / (data['{} UB'.format(data.columns.values[i])] - data['{} LB'.format(data.columns.values[i])])
    return data

def calMACD(symbols, start_date, end_date, low_period, high_period, signal):
    data = get_data([symbols], pd.date_range(start_date, end_date))
    data = data[[symbols]]
    data = data.iloc[:]/data.iloc[0]
    data['low_ema'] = pd.ewma(data[symbols], span = low_period)
    data['high_ema'] = pd.ewma(data[symbols], span = high_period)
    data['MACD'] = data['low_ema'] - data['high_ema']
    data['signal'] = pd.ewma(data['MACD'], span = signal)
    return data

def calOBV(symbols, start_date, end_date):
    data = get_data([symbols], pd.date_range(start_date, end_date))
    data = data[[symbols]]
    volumn = get_data([symbols], pd.date_range(start_date, end_date), colname = 'Volume')
    volumn = volumn[[symbols]]
    exm = data.copy()
    for i in range(exm.shape[0] - 1,0,-1):
	if exm.iloc[i].values > exm.iloc[i-1].values:
	    exm.iloc[i] = 1
	elif exm.iloc[i].values < exm.iloc[i-1].values:
	    exm.iloc[i] = -1
        else:
	    exm.iloc[i] = 0
    exm.iloc[0] = 0
    volumn = exm * volumn
    for i in range(volumn.shape[0] - 1):
	volumn.iloc[i+1] = volumn.iloc[i+1] + volumn.iloc[i]
    return volumn
def test_code():
    """Change the symbol here"""
    symbols = 'JPM'
    """Call different indicators method here"""
    OBV = KDJ(symbols, start_date = dt.datetime(2008,1,1), end_date = dt.datetime(2009,12,31), n = 9, a = 3, b = 3)
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = OBV.plot(title='On-balance Volume', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    """save the picture of the indicators"""
    plt.savefig('OBV.png')
if __name__ == "__main__":
    test_code()
