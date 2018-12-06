"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    orderfile = pd.read_csv(orders_file)
    start_date = orderfile['Date'].iloc[0]
    end_date = orderfile['Date'].iloc[-1]
    price = get_data(list(set(orderfile['Symbol'])), pd.date_range(start_date, end_date))
    price = price[list(set(orderfile['Symbol']))]  # remove SPY
    price['CASH'] = pd.Series(np.ones(price.shape[0]), index = price.index)
    trade = price.copy()
    trade.iloc[:] = 0
    for index, row in orderfile.iterrows():
	if row['Order'] == 'BUY':
	    symbol = row['Symbol']
	    date = row['Date']
    	    trade.loc[trade.index == date, symbol] += row['Shares']
	    trade.loc[trade.index == date, 'CASH'] -= (row['Shares'] * (price.loc[price.index == date, symbol])*(1+impact))
	    trade.loc[trade.index == date, 'CASH'] -= commission
        elif row['Order'] == 'SELL':
	    symbol = row['Symbol']
	    date = row['Date']
    	    trade.loc[trade.index == date, symbol] -= row['Shares']
	    trade.loc[trade.index == date, 'CASH'] += row['Shares'] * (price.loc[price.index == date, symbol]*(1-impact))
	    trade.loc[trade.index == date, 'CASH'] -= commission
    holding = trade.copy()
    holding['CASH'].iloc[0] += start_val
    for i in range(holding.shape[0] - 1):
	holding.iloc[i+1] += holding.iloc[i]
    portvals = holding*price.values
    portvals['TOTAL'] = pd.Series(np.zeros(price.shape[0]), index = price.index)
    portvals['TOTAL'] = portvals.sum(axis = 1)
    answer = portvals.filter(['TOTAL'], axis = 1)
    return answer

def author():
    return 'cfeng46'

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv, commission = 0, impact = 0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    print(portvals)
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    #print "Date Range: {} to {}".format(start_date, end_date)
    #print
    #print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    #print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    #print
    #print "Cumulative Return of Fund: {}".format(cum_ret)
    #print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    #print
    #print "Standard Deviation of Fund: {}".format(std_daily_ret)
    #print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    #print
    #print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    #print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    #print
    #print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
