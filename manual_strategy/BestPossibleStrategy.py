import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data
import math
def testPolicy(symbols, sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000):
    data = get_data([symbols], pd.date_range(sd, ed))
    data = data[[symbols]]
    order = data.copy()
    order.columns = ['Symbol']
    order['Order'] = np.zeros(order.shape[0])
    order['Share'] = np.zeros(order.shape[0])
    order['Symbol'] = symbols
    order['CUM'] = np.zeros(order.shape[0])
    for i in range(data.shape[0]-1):
	if data.iloc[i+1].values > data.iloc[i].values:
	    order['CUM'].iloc[i] = 1000
        elif data.iloc[i+1].values < data.iloc[i].values:
	    order['CUM'].iloc[i] = -1000
	else:
	    order['CUM'].iloc[i] = 0
    for i in range(order.shape[0]):
	if order['CUM'].iloc[i] < order['CUM'].iloc[i-1]:
	    order['Order'].iloc[i] = 'SELL'
	    order['Share'].iloc[i] = abs(order['CUM'].iloc[i] - order['CUM'].iloc[i-1])
	elif order['CUM'].iloc[i] > order['CUM'].iloc[i-1]:
	    order['Order'].iloc[i] = 'BUY'
	    order['Share'].iloc[i] = abs(order['CUM'].iloc[i] - order['CUM'].iloc[i-1])
    #order = order.drop(order[order['Order'] == 0].index)
    answer = order.iloc[:,:-1]
    return answer
	    
	
if __name__ == "__main__":
    result = testPolicy(symbols = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), sv = 100000)
    
