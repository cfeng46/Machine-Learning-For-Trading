import pandas as pd
import numpy as np
import datetime as dt
import indicators as indic
import os
from util import get_data, plot_data
import math

#Chengliang Feng
#cfeng46

def testPolicy(symbols, sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000):
    sma = indic.calSMA(symbols, start_date = sd, end_date = ed, period = 20).fillna(method = 'bfill')
    BB = indic.calBB(symbols, start_date = sd, end_date = ed, period = 20).fillna(method = 'bfill')
    MACD = indic.calMACD(symbols, start_date = sd, end_date = ed, low_period = 12, high_period = 26, signal = 9).fillna(method = 'bfill')
    OBV = indic.calOBV(symbols, start_date = sd, end_date = ed)
    order = sma.copy()
    order.columns = ['Symbol', 'Order', 'Share']
    order.iloc[:] = 0
    order['Symbol'] = symbols
    order['CUM'] = np.zeros(order.shape[0])
    order['CUM'].iloc[:] = np.NAN
    for i in range(sma.shape[0]):
	#if (np.isnan(sma['{} price/SMA ratio'.format(symbols)].iloc[i]) or np.isnan(BB['{} BBP'.format(symbols)].iloc[i]) or np.isnan(MACD['MACD'].iloc[i])) == False:
	if OBV[symbols].iloc[i] - OBV[symbols].iloc[i-1] < 0 and sma['{} price/SMA ratio'.format(symbols)].iloc[i] < 0.8 and BB['{} BBP'.format(symbols)].iloc[i] < 0.05 and BB['{} BBP'.format(symbols)].iloc[i] > 0 and MACD['MACD'].iloc[i] < MACD['signal'].iloc[i] and MACD['MACD'].iloc[i] < 0 and MACD['signal'].iloc[i] < 0:
	    order['CUM'].iloc[i] = 1000
	elif OBV[symbols].iloc[i] - OBV[symbols].iloc[i-1] > 0 and sma['{} price/SMA ratio'.format(symbols)].iloc[i] > 1.05 and BB['{} BBP'.format(symbols)].iloc[i] > 1 and MACD['MACD'].iloc[i] > MACD['signal'].iloc[i] and MACD['MACD'].iloc[i] > 0 and MACD['signal'].iloc[i] > 0:
	    order['CUM'].iloc[i] = -1000
    order.fillna(method = "ffill", inplace = True)
    order.fillna(0, inplace = True)
    for i in range(1, order.shape[0]):
	if order['CUM'].iloc[i] < order['CUM'].iloc[i-1]:
	    order['Order'].iloc[i] = 'SELL'
	    order['Share'].iloc[i] = abs(order['CUM'].iloc[i] - order['CUM'].iloc[i-1])
	elif order['CUM'].iloc[i] > order['CUM'].iloc[i-1]:
	    order['Order'].iloc[i] = 'BUY'
	    order['Share'].iloc[i] = abs(order['CUM'].iloc[i] - order['CUM'].iloc[i-1])
    answer = order.iloc[:,:-1]
    return answer

if __name__=='__main__':
    testPolicy(symbols = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000)
