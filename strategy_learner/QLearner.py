"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""
#Chengliang Feng
#cfeng46

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
	self.num_states = num_states
	self.alpha = alpha
	self.gamma = gamma
	self.rar = rar
	self.radr = radr
	self.dyna = dyna
	self.Q = np.random.random_sample((num_states, num_actions))
        self.s = 0
        self.a = 0
	self.stateExp = []
	self.actionExp = []
	if self.dyna != 0:
	    self.TC = np.ones((num_states, num_actions, num_states))
	    self.TC.fill(0.00001)
	    self.T = self.TC/self.TC.sum(axis = 2, keepdims = True)
	    self.R = np.random.random_sample((num_states, num_actions))
	    
    def author(self):
	return 'cfeng46'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
	if rand.random() > self.rar:
	    action = np.argmax(self.Q[s,:])
	else:
	    action = rand.randint(0, self.num_actions - 1)
	self.stateExp.append(self.s)
	self.actionExp.append(self.a)
	self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha*(r + self.gamma * np.max(self.Q[s_prime, :]))
	self.stateExp.append(self.s)
	self.actionExp.append(self.a)
	if self.dyna > 0:
	    self.TC[self.s, self.a, s_prime] += 1
	    self.T[self.s, self.a, :] = self.TC[self.s, self.a, :] / self.TC[self.s, self.a, :].sum()
	    self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r
	    for i in range(self.dyna):
		s_sim = rand.choice(self.stateExp)
	 	a_sim = rand.choice(self.actionExp)
		s_primeS = self.T[s_sim, a_sim, :].argmax()
	        r_s = self.R[s_sim, a_sim]
	        self.Q[s_sim, a_sim] = (1 - self.alpha) * self.Q[s_sim, a_sim] + self.alpha*(r_s + self.gamma * np.max(self.Q[s_primeS, :]))
	if rand.random() > self.rar:
	    action = np.argmax(self.Q[s_prime,:])
	else:
	    action = rand.randint(0, self.num_actions - 1)
	self.s = s_prime
	self.a = action
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return self.a

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
