"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import math

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def f(x, args):
    x = np.asarray(x)
    pos_val = args * x
    port_val = pos_val.sum(axis = 1)
    dr = (port_val/port_val.shift(1)) - 1
    dr[0] = 1
    sddr = dr[1:].std()
    return sddr
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    #allocs = np.asarray([0.2, 0.2, 0.3, 0.3]) # add code here to find the allocations
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Get daily portfolio value
    #port_val = prices_SPY # add code here to compute daily portfolio values
    prices.fillna(method = "ffill", inplace = True)
    prices.fillna(method = "bfill", inplace = True)
    norm = prices/prices.iloc[0]
    allocs = list(1/(len(syms)*1.0) for x in syms)
    bnds = tuple((0.0,1.0) for x in allocs)
    cons = ({'type':'eq', "fun": lambda x: 1-np.sum(x)})
    min_result = spo.minimize(f, allocs, args=norm, method = 'SLSQP', bounds = bnds, constraints = cons, options={'disp':True})
    allocs = min_result.x
    pos_val = norm*allocs
    port_val = pos_val.sum(axis = 1)
    dr = (port_val/port_val.shift(1)) - 1
    dr[0] = 1
    adr = dr[1:].mean()
    cr = (port_val[-1]/port_val[0]) - 1
    sddr = dr[1:].std()
    sr = ((adr)/sddr) * math.sqrt(252)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
	prices_SPY = prices_SPY/prices_SPY[0]
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot(title = "Daily Portfolio Value and SPY")
        ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.savefig('plot.png')

    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
