"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
#Chengliang Feng
#cfeng46

import datetime as dt
import pandas as pd
import util as ut
import random
import QLearner as ql
import numpy as np
from indicators import *
import time
import ManualStrategy as ms
import marketsimcode as market

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 100000): 

        # add your code to do learning here
	self.learner = ql.QLearner(num_states = 3000000, num_actions = 3, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False)

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices
  
        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume
	train_SMA_ratio = calSMA(symbol, sd, ed, 20).ix[:, -1:]
	train_bbp = calBB(symbol, sd, ed, 20).ix[:, -1:].fillna(method = 'bfill')
	train_OBV = calOBV(symbol, sd, ed)
	#print train_OBV
	train_OBV = train_OBV - train_OBV.shift(1)
	train_MACD = calMACD(symbol, sd, ed, 12, 26, 9).ix[:, -2:-1]
	train_SIG = calMACD(symbol, sd, ed, 12, 26, 9).ix[:, -1:]
	train_diff = pd.concat([train_MACD, train_SIG*-1],axis = 1)
	train_diff['diff'] = train_diff.sum(axis = 1)
	train_ind = train_diff.iloc[:, -1:]
	#print train_OBV
	"Discretize"
	bin_train_SMA_ratio = np.linspace(train_SMA_ratio.min(), train_SMA_ratio.max(), 10)
	train_SMA_ratio.iloc[:,0] = np.digitize(train_SMA_ratio, bin_train_SMA_ratio) - 1

	bin_train_bbp = np.linspace(train_bbp.min(), train_bbp.max(), 10)
	train_bbp.iloc[:,0] = np.digitize(train_bbp, bin_train_bbp) - 1

	bin_train_OBV = np.linspace(train_OBV.min(), train_OBV.max(), 10)
	train_OBV.iloc[:,0] = np.digitize(train_OBV, bin_train_OBV) - 1

	bin_train_ind = np.linspace(train_ind.min(), train_ind.max(), 10)
	train_ind.iloc[:,0] = np.digitize(train_ind, bin_train_ind) - 1

	bin_train_MACD = np.linspace(train_MACD.min(), train_MACD.max(), 10)
	train_MACD.iloc[:,0] = np.digitize(train_MACD, bin_train_MACD) - 1

	bin_train_SIG = np.linspace(train_SIG.min(), train_SIG.max(), 10)
	train_SIG.iloc[:,0] = np.digitize(train_SIG, bin_train_SIG) - 1

	train_signal = pd.concat([train_bbp*1000, train_OBV*10000, train_SMA_ratio*100000, train_ind*100, train_MACD, train_SIG*10], axis = 1)
	train_signal['state'] = train_signal.sum(axis = 1)
	train_states = train_signal.iloc[:, -1:]
	Q_table = train_signal.iloc[:, :-3].copy()
        Q_table.columns=['Pos', 'Price', 'Cash', 'PV']
        train_states = train_states.values
        Q_table.ix[:, 'Pos'] = 0
        Q_table.ix[:, 'Price'] = prices.ix[:, symbol]
        Q_table.ix[:, 'Cash'] = Q_table.ix[:, 'PV'] = sv
        Q_table = Q_table.values
	stop = 0
        iteration = 0
	cum = 0
        while iteration < 1500 and stop < 5:
            days = 0
            position = 0
            state = position * 100000 + train_states[days, 0]
            action = self.learner.querysetstate(state)

            for days in range(1, train_states.size):
                if position == 0:
                    if action == 1:
                        position = 1
                        Q_table[days, 0] = -1000
                        Q_table[days, 2] = Q_table[days - 1, 2] + Q_table[days, 1] * 1000
                    elif action == 2:
                        position = 2
                        Q_table[days, 0] = 1000
                        Q_table[days, 2] = Q_table[days - 1, 2] - Q_table[days, 1] * 1000
                    else:
                        position = 0
                        Q_table[days, 0] = Q_table[days - 1, 0]
                        Q_table[days, 2] = Q_table[days - 1, 2]

                elif position == 1 and action == 2:
                    position = 2
                    Q_table[days, 0] = 1000
                    Q_table[days, 2] = Q_table[days - 1, 2] - Q_table[days, 1] * 2000

                elif position == 2 and action == 1:
                    position = 1
                    Q_table[days, 0] = -1000
                    Q_table[days, 2] = Q_table[days - 1, 2] + Q_table[days, 1] * 2000

                else:
                    position = position
                    Q_table[days, 0] = Q_table[days - 1, 0]
                    Q_table[days, 2] = Q_table[days - 1, 2]

                Q_table[days, 3] = Q_table[days, 2] + Q_table[days, 0] * Q_table[days, 1]
                reward = Q_table[days, 3] / Q_table[days - 1, 3] - 1 - 2 * self.impact
                state = position * 100000 + train_states[days, 0]
                action = self.learner.query(state, reward)
	    ret = Q_table[-1, 3] - sv / sv
	    if cum == ret:
		stop += 1
	    else:
		stop = 0
            iteration += 1
	    
	    
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
	

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
	trade_frame = prices.copy()
	trade_frame.columns = ['Symbol']
	trade_frame['Order'] = np.zeros(trade_frame.shape[0])
	trade_frame['Share'] = np.zeros(trade_frame.shape[0])
    	trade_frame.iloc[:] = 0
    	trade_frame['Symbol'] = symbol
        prices_SPY = prices_all['SPY']  # only SPY, for comparison latery

	train_SMA_ratio = calSMA(symbol, sd, ed, 20).ix[:, -1:]
	train_bbp = calBB(symbol, sd, ed, 20).ix[:, -1:].fillna(method = 'bfill')
	train_OBV = calOBV(symbol, sd, ed)
	#print train_OBV
	train_OBV = train_OBV - train_OBV.shift(1)
	train_MACD = calMACD(symbol, sd, ed, 12, 26, 9).ix[:, -2:-1]
	train_SIG = calMACD(symbol, sd, ed, 12, 26, 9).ix[:, -1:]
	train_diff = pd.concat([train_MACD, train_SIG*-1],axis = 1)
	train_diff['diff'] = train_diff.sum(axis = 1)
	train_ind = train_diff.iloc[:, -1:]
	#print train_OBV
	"Discretize"
	bin_train_SMA_ratio = np.linspace(train_SMA_ratio.min(), train_SMA_ratio.max(), 10)
	train_SMA_ratio.iloc[:,0] = np.digitize(train_SMA_ratio, bin_train_SMA_ratio) - 1

	bin_train_bbp = np.linspace(train_bbp.min(), train_bbp.max(), 10)
	train_bbp.iloc[:,0] = np.digitize(train_bbp, bin_train_bbp) - 1

	bin_train_OBV = np.linspace(train_OBV.min(), train_OBV.max(), 10)
	train_OBV.iloc[:,0] = np.digitize(train_OBV, bin_train_OBV) - 1

	bin_train_ind = np.linspace(train_ind.min(), train_ind.max(), 10)
	train_ind.iloc[:,0] = np.digitize(train_ind, bin_train_ind) - 1

	bin_train_MACD = np.linspace(train_MACD.min(), train_MACD.max(), 10)
	train_MACD.iloc[:,0] = np.digitize(train_MACD, bin_train_MACD) - 1

	bin_train_SIG = np.linspace(train_SIG.min(), train_SIG.max(), 10)
	train_SIG.iloc[:,0] = np.digitize(train_SIG, bin_train_SIG) - 1

	train_signal = pd.concat([train_bbp*1000, train_OBV*10000, train_SMA_ratio*100000, train_ind*100, train_MACD, train_SIG*10], axis = 1)
	train_signal['state'] = train_signal.sum(axis = 1)
	test_states = train_signal.iloc[:, -1:]

        '''TEST'''
        test_states = test_states.values
        position = 0

        for days in range(1, test_states.size):
            state = position * 100000 + test_states[days-1, 0]
            action = self.learner.querysetstate(state)
            if position == 0:
                if action == 1:
                    trade_frame['Share'].iloc[days] = -1000
                    position = 1
                elif action == 2:
                    trade_frame['Share'].iloc[days] = 1000
                    position = 2
                else:
                    position = 0

            elif position == 1 and action == 2:
                trade_frame['Share'].iloc[days] = 2000
                position = 2

            elif position == 2 and action == 1:
                trade_frame['Share'].iloc[days] = -2000
                position = 1

            else:
                position = position
		trade_frame['Share'].iloc[days] = trade_frame['Share'].iloc[days - 1]
    	for i in range(1, trade_frame.shape[0]):
	    if trade_frame['Share'].iloc[i] < 0:
		trade_frame['Order'].iloc[i] = 'SELL'
	        trade_frame['Share'].iloc[i] = abs(trade_frame['Share'].iloc[i])
	    elif trade_frame['Share'].iloc[i] > 0:
		trade_frame['Order'].iloc[i] = 'BUY'
		trade_frame['Share'].iloc[i] = abs(trade_frame['Share'].iloc[i])
        if self.verbose: print type(trade) # it better be a DataFrame!
        if self.verbose: print trade
        if self.verbose: print prices_all
        return trade_frame

def test_code():
    sv = 100000
    impact = 0
    np.random.seed(1481090000)
    random.seed(1481090000)
    learner = StrategyLearner(verbose = False, impact = 0.000)
    learner.addEvidence(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    learner_trade = learner.testPolicy(symbol = 'UNH', sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), sv = 100000)
    learner_val, BM_port = market.compute_portvals(orderfile = learner_trade, start_val = sv, commission = 0, impact = impact)
    learner_val.columns = ["learner_Total"]
    print learner_val, BM_port

if __name__ == '__main__':
    test_code()















