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

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=100000):

        self.learner = ql.QLearner(num_states=3000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)  # initialize the learner
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume


        train_momentum = (prices / prices.shift(15) - 1).fillna(method='bfill')
        train_bbp = calBB(symbol, sd, ed, 20).ix[:, -1:].fillna(method = 'bfill')
        train_daily_rets = (prices / prices.shift(1)) - 1
        train_vol = pd.rolling_std(train_daily_rets, 15).fillna(method='bfill')

        '''DISCRETIZE'''
        bins_bbp = np.linspace(train_bbp.ix[:, 0].min(), train_bbp.ix[:, 0].max(), 10)
        train_bbp.ix[:, 0] = np.digitize(train_bbp.ix[:, 0], bins_bbp) - 1

        bins_momentum = np.linspace(train_momentum.ix[:, 0].min(), train_momentum.ix[:, 0].max(), 10)
        train_momentum.ix[:, 0] = np.digitize(train_momentum.ix[:, 0], bins_momentum) - 1
        
        bins_vol = np.linspace(train_vol.ix[:, 0].min(), train_vol.ix[:, 0].max(), 10)
        train_vol.ix[:, 0] = np.digitize(train_vol.ix[:, 0], bins_vol) - 1

        train_signal = pd.concat([train_bbp*100, train_momentum*10, train_vol], axis=1)
	train_signal['state'] = train_signal.sum(axis = 1)
	train_states = train_signal.iloc[:, -1:]
        print train_signal
	Q_table = train_signal.copy()
        Q_table.columns=['Pos', 'Price', 'Cash', 'PV']
        train_states = train_states.values
        Q_table.ix[:, 'Pos'] = 0
        Q_table.ix[:, 'Price'] = prices.ix[:, symbol]
        Q_table.ix[:, 'Cash'] = Q_table.ix[:, 'PV'] = sv
        Q_table = Q_table.values
        print Q_table
	stop = 0
        iteration = 0
	cum = 0
        while iteration < 1500 and stop < 5:
            days = 0
            position = 0
            state = position * 1000 + train_states[days, 0]
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
                state = position * 1000 + train_states[days, 0]
                action = self.learner.query(state, reward)
	    ret = Q_table[-1, 3] - sv / sv
	    if cum == ret:
		stop += 1
	    else:
		stop = 0
            iteration += 1
        print pd.DataFrame(Q_table)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol]]
        trade_frame = prices.copy()
	trade_frame.columns = ['Share']
    	trade_frame.iloc[:] = 0
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

       
        test_momentum = (prices / prices.shift(15) - 1).fillna(method='bfill')
        test_bbp = calBB(symbol, sd, ed, 20).ix[:, -1:].fillna(method = 'bfill')
        test_daily_rets = (prices / prices.shift(1)) - 1
        test_vol = pd.rolling_std(test_daily_rets, 15).fillna(method='bfill')



        '''DISCRETIZE'''
        bins_bbp = np.linspace(test_bbp.ix[:, 0].min(), test_bbp.ix[:, 0].max(), 10)
        test_bbp.ix[:, 0] = np.digitize(test_bbp.ix[:, 0], bins_bbp) - 1

        bins_momentum = np.linspace(test_momentum.ix[:, 0].min(), test_momentum.ix[:, 0].max(), 10)
        test_momentum.ix[:, 0] = np.digitize(test_momentum.ix[:, 0], bins_momentum) - 1
        
        bins_vol = np.linspace(test_vol.ix[:, 0].min(), test_vol.ix[:, 0].max(), 10)
        test_vol.ix[:, 0] = np.digitize(test_vol.ix[:, 0], bins_vol) - 1

        test_signal = pd.concat([test_bbp*100, test_momentum*10, test_vol], axis=1)
	test_signal['state'] = test_signal.sum(axis = 1)
	test_states = test_signal.iloc[:, -1:]

        '''TEST'''
        test_states = test_states.values
        position = 0

        for days in range(1, test_states.size):
            state = position * 1000 + test_states[days-1, 0]
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

        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trade_frame

if __name__=="__main__":
    learner = StrategyLearner(verbose = False, impact = 0.000)
    learner.addEvidence(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    trade = learner.testPolicy(symbol="UNH",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    print trade
    trade1 = learner.testPolicy(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    print trade1
    print "One does not simply think up a strategy"
