"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import BestPossibleStrategy as bps
import os
from util import get_data, plot_data
import math
import ManualStrategy as ms

def compute_portvals(orderfile, start_val = 1000000, commission = 0, impact = 0):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    #orderfile = bps.testPolicy(symbol = 'JPM', sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), star = start_val)
    start_date = orderfile.index.values[0]
    end_date = orderfile.index.values[-1]
    symbol = list(set(orderfile['Symbol']))
    print symbol
    price_all = get_data(list(set(orderfile['Symbol'])), pd.date_range(start_date, end_date))
    price = price_all[list(set(orderfile['Symbol']))]  # remove SPY
    price['CASH'] = pd.Series(np.ones(price.shape[0]), index = price.index)
    trade = price.copy()
    BM = price.copy()
    BM_holding = price.copy()
    BM_holding.iloc[:] = 0
    BM_holding[symbol[0]].iloc[0] = 1000
    BM_holding['CASH'].iloc[0] -= BM[symbol[0]].iloc[0] * 1000*(1 + impact) + commission
    BM_holding[symbol[0]].iloc[-1] = -1000
    BM_holding['CASH'].iloc[-1] += BM[symbol[0]].iloc[-1] * 1000*(1-impact) - commission
    trade.iloc[:] = 0
    for i in range(orderfile.shape[0]):
	if orderfile['Order'].iloc[i] == 'BUY':
	    symbol = orderfile['Symbol'].iloc[i]
	    date = orderfile.index.values[i]
    	    trade.loc[trade.index == date, symbol] += orderfile['Share'].iloc[i]
	    trade.loc[trade.index == date, 'CASH'] -= (orderfile['Share'].iloc[i] * (price.loc[price.index == date, symbol])*(1+impact))
	    trade.loc[trade.index == date, 'CASH'] -= commission
        elif orderfile['Order'].iloc[i] == 'SELL':
	    symbol = orderfile['Symbol'].iloc[i]
	    date = orderfile.index.values[i]
    	    trade.loc[trade.index == date, symbol] -= orderfile['Share'].iloc[i]
	    trade.loc[trade.index == date, 'CASH'] += orderfile['Share'].iloc[i] * (price.loc[price.index == date, symbol]*(1-impact))
	    trade.loc[trade.index == date, 'CASH'] -= commission
    holding = trade.copy()
    BM_holding['CASH'].iloc[0] += start_val
    holding['CASH'].iloc[0] += start_val
    for i in range(holding.shape[0] - 1):
	holding.iloc[i+1] += holding.iloc[i]
	BM_holding.iloc[i+1] += BM_holding.iloc[i]
    portvals = holding*price.values
    BM_port = BM_holding*BM.values
    portvals['TOTAL'] = pd.Series(np.zeros(price.shape[0]), index = price.index)
    BM_port['BM_TOTAL'] = pd.Series(np.zeros(BM.shape[0]), index = BM.index)
    portvals['TOTAL'] = portvals.sum(axis = 1)
    BM_port['BM_TOTAL'] = BM_port.sum(axis = 1)
    answer = portvals.filter(['TOTAL'], axis = 1)
    BM_answer = BM_port.filter(['BM_TOTAL'], axis = 1)
    return answer, BM_answer

def author():
    return 'cfeng46'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    """"Call different testPolicy here"""
    df_trade = ms.testPolicy(symbols = 'AAPL', sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), sv = 100000)
    sv = 100000

    # Process orders
    #if you are using best possible strategy, change the commission and impact to 0
    port_val, BM_port = compute_portvals(orderfile = df_trade, start_val = sv, commission = 9.95, impact = 0.05)
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    if isinstance(BM_port, pd.DataFrame):
        BM_port = BM_port[BM_port.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    dr = (port_val/port_val.shift(1)) - 1
    dr[0] = 0
    adr = dr[1:].mean()
    cr = (port_val[-1]/port_val[0]) - 1
    sddr = dr[1:].std()
    sr = ((adr)/sddr) * math.sqrt(252)
    ev = port_val[-1]
    BM_dr = (BM_port/BM_port.shift(1)) - 1
    BM_dr[0] = 0
    BM_adr = BM_dr[1:].mean()
    BM_cr = (BM_port[-1]/BM_port[0]) - 1
    BM_sddr = BM_dr[1:].std()
    BM_sr = ((BM_adr)/BM_sddr) * math.sqrt(252)
    BM_ev = BM_port[-1]
    show_BM = BM_port/BM_port[0]
    show_port = port_val/port_val[0]
    answer = pd.concat([show_BM, show_port], axis = 1)
    """comment the below block of lines if using best possible strategy"""
    chart = df_trade.drop(df_trade[df_trade['Order'] == 0].index)
    long_date = []
    short_date = []
    if chart['Order'].iloc[0] == 'BUY':
	long_date.append(chart.index.values[0])
    elif chart['Order'].iloc[0] == 'SELL':
	short_date.append(chart.index.values[0])
    for i in range(chart.shape[0]):
	if chart['Share'].iloc[i] > 1000 and chart['Order'].iloc[i] == 'BUY':
	    long_date.append(chart.index.values[i])
        elif chart['Share'].iloc[i] > 1000 and chart['Order'].iloc[i] == 'SELL':
	    short_date.append(chart.index.values[i])
    """comment the above block of lines if using best possible strategy"""
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = answer.plot(title='port value', fontsize=12, color = ['b', 'k'])
    ax.set_xlabel('Date')
    ax.set_ylabel('normalized Port Values')
    for point in long_date:
	ax.axvline(x = point, color = 'g')
    for exit in short_date:
	ax.axvline(x = exit, color = 'r')
    plt.savefig('outSample.png')

    print
    print "Sharpe Ratio of Fund: {}".format(sr)
    print "Sharpe Ratio of BM : {}".format(BM_sr)
    print
    print "Cumulative Return of Fund: {}".format(cr)
    print "Cumulative Return of BM : {}".format(BM_cr)
    print
    print "Standard Deviation of Fund: {}".format(sddr)
    print "Standard Deviation of BM : {}".format(BM_sddr)
    print
    print "Average Daily Return of Fund: {}".format(adr)
    print "Average Daily Return of BM : {}".format(BM_adr)
    print
    print "Final Portfolio Value: {}".format(port_val[-1])
    print
    print "Final BM_Portfolio Value: {}".format(BM_port[-1])

if __name__ == "__main__":
    test_code()
