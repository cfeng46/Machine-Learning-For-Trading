"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.randint(20, size=(50,25))
    answer = []
    j = 1
    for i in range(X.shape[1]):
        num = X[:,i]*j
	answer.append(num)
	j += 1
    Y = sum(answer)
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    r = np.random.rand(20,)
    Y = np.random.randint(20, size = (50,))
    y = (Y-Y.mean())/Y.std()
    x = sum(y)*r
    X = np.random.normal(0, 1, (50,20))
    for i in range(X.shape[1]):
	X[:,i] = (X[:,i]-X[:,i].mean())/X[:,i].std()
	X[:,i] -= (sum(X[:,i])-x[i])/X.shape[0]
    return X, Y

def author():
    return 'cfeng46' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
